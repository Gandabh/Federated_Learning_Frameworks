import tensorflow as tf
import tensorflow_federated as tff
import time
import psutil
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create client datasets
def create_tf_dataset_for_client(x, y, num_clients):
    num_examples_per_client = len(x) // num_clients
    client_data = []
    for i in range(num_clients):
        start = i * num_examples_per_client
        end = (i + 1) * num_examples_per_client
        client_data.append(
            tf.data.Dataset.from_tensor_slices((x[start:end], y[start:end]))
            .batch(32)
        )
    return client_data

# Define the model
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Federated Averaging process
def federated_averaging(num_clients, num_rounds, num_epochs):
    train_data = create_tf_dataset_for_client(x_train, y_train, num_clients)
    
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    )
    
    state = trainer.initialize()
    
    start_time = time.time()
    
    for round_num in range(num_rounds):
        sampled_clients = train_data
        
        for _ in range(num_epochs):
            state, metrics = trainer.next(state, sampled_clients)
        
        if (round_num + 1) % 1 == 0:
            print(f'Round {round_num + 1}, Metrics: {metrics}')
            print(f'CPU Usage: {psutil.cpu_percent()}%')
            print(f'RAM Usage: {psutil.virtual_memory().percent}%')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return metrics, total_time

# Run experiments with different numbers of clients
client_counts = [1, 10, 50, 100]
num_rounds = 10
num_epochs = 15

results = {}

for num_clients in client_counts:
    print(f"\nRunning experiment with {num_clients} clients")
    metrics, total_time = federated_averaging(num_clients, num_rounds, num_epochs)
    results[num_clients] = {
        'accuracy': metrics['train']['sparse_categorical_accuracy'],
        'loss': metrics['train']['loss'],
        'training_time': total_time
    }

# final results
for num_clients, result in results.items():
    print(f"\nResults for {num_clients} clients:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Training Time: {result['training_time']:.2f} seconds")
