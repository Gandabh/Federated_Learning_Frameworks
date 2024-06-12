# Import necessary libraries
from substra import Client
from substra.sdk.schemas import DatasetSpec, Permissions, DataSampleSpec
from substrafl.index_generator import NpIndexGenerator
from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.strategies import FedAvg
from substrafl.nodes import TrainDataNode, AggregationNode, TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import substrafl
import os

# Define constants
N_CLIENTS = 3
NUM_UPDATES = 100
BATCH_SIZE = 32
NUM_ROUNDS = 10  # Increased the number of rounds
SEED = 42

# Setup clients
client_0 = Client(client_name="org-1")
client_1 = Client(client_name="org-2")
client_2 = Client(client_name="org-3")

clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}

ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]

# Download CIFAR-10 dataset and split data for each organization
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split data for organizations
def split_dataset(dataset, num_clients):
    length = len(dataset) // num_clients
    return [torch.utils.data.Subset(dataset, list(range(i*length, (i+1)*length))) for i in range(num_clients)]

train_subsets = split_dataset(trainset, len(DATA_PROVIDER_ORGS_ID))
test_subsets = split_dataset(testset, len(DATA_PROVIDER_ORGS_ID))

# Save subsets to local directories
data_path = pathlib.Path.cwd() / "tmp" / "data_cifar10"
data_path.mkdir(parents=True, exist_ok=True)

for i, (train_subset, test_subset) in enumerate(zip(train_subsets, test_subsets)):
    org_path = data_path / f"org_{i+1}"
    train_path = org_path / "train"
    test_path = org_path / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)
    
    torch.save(next(iter(train_loader)), train_path / "data.pt")
    torch.save(next(iter(test_loader)), test_path / "data.pt")

# Register datasets and datasamples
assets_directory = pathlib.Path.cwd() / "torch_fedavg_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]
    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    dataset = DatasetSpec(
        name="CIFAR-10",
        data_opener=assets_directory / "dataset" / "cifar10_opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)

    train_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=train_path,
    )
    train_datasample_keys[org_id] = client.add_data_sample(train_sample)

    test_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=test_path,
    )
    test_datasample_keys[org_id] = client.add_data_sample(test_sample)

# Metrics functions
def accuracy(data_from_opener, predictions):
    y_true = data_from_opener["labels"]
    return accuracy_score(y_true, np.argmax(predictions, axis=1))

def roc_auc(data_from_opener, predictions):
    y_true = data_from_opener["labels"]
    n_class = np.max(y_true) + 1
    y_true_one_hot = np.eye(n_class)[y_true]
    return roc_auc_score(y_true_one_hot, predictions)

# Define model
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * 256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x, eval=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 4 * 4 * 256)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=not eval)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CIFAR10CNN()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adjusted learning rate
criterion = nn.CrossEntropyLoss()

# Define index generator
index_generator = NpIndexGenerator(batch_size=BATCH_SIZE, num_updates=500)  # Increased number of updates

# Define TorchDataset
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data_from_opener, is_inference: bool):
        self.x = data_from_opener["images"]
        self.y = data_from_opener["labels"]
        self.is_inference = is_inference

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx]) / 255
        if self.is_inference:
            return x
        y = torch.tensor(self.y[idx]).type(torch.int64)
        return x, y

    def __len__(self):
        return len(self.x)

# Define SubstraFL algorithm
class TorchCNN(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=SEED,
            use_gpu=False,
        )

# Define federated learning strategy
strategy = FedAvg(algo=TorchCNN(), metric_functions={"Accuracy": accuracy, "ROC AUC": roc_auc})

# Define nodes
aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = [
    TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]

test_data_nodes = [
    TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[test_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]

eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)

# Define dependencies
dependencies = Dependency(pypi_dependencies=["numpy==1.24.3", "scikit-learn==1.3.1", "torch==2.0.1", "torchvision==0.15.1", "--extra-index-url https://download.pytorch.org/whl/cpu"])

# Execute experiment
substrafl.set_logging_level(loglevel=logging.ERROR)

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
    name="CIFAR-10 documentation example",
)

# Wait for the compute plan to finish
client_0.wait_compute_plan(compute_plan.key)

# Display results
performances_df = pd.DataFrame(client_0.get_performances(compute_plan.key).model_dump())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "identifier", "performance"]])

# Display results
performances_df = pd.DataFrame(client_0.get_performances(compute_plan.key).model_dump())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "identifier", "performance"]])

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Test dataset results")

axs[0].set_title("Accuracy")
axs[1].set_title("ROC AUC")

for ax in axs.flat:
    ax.set(xlabel="Rounds", ylabel="Score")

# Group the data by organization and identifier (metric)
grouped = performances_df.groupby(['worker', 'identifier'])

for name, group in grouped:
    worker, metric = name
    if metric == 'Accuracy':
        axs[0].plot(group['round_idx'], group['performance'], label=f'{worker} - {metric}')
    elif metric == 'ROC AUC':
        axs[1].plot(group['round_idx'], group['performance'], label=f'{worker} - {metric}')

# Add legends
axs[0].legend(loc='lower right')
axs[1].legend(loc='lower right')

# Save the plot as an image file
plt.savefig('results_plot.png')

# Show the plot
plt.show()


# Download model
from substrafl.model_loading import download_algo_state

client_to_download_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = None

algo = download_algo_state(
    client=clients[client_to_download_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
)

model = algo.model
print(model)
