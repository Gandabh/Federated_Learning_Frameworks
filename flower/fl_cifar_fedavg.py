import flwr as fl
import tensorflow as tf
import numpy as np

# Function to load CIFAR-10
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = load_cifar10()

# Define the model
def create_compiled_model():
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_compiled_model()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(x_train, y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(x_train)

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower server
fl.server.start_server(
    "127.0.0.1:8080",
    config={"num_rounds": 3},
    strategy=fl.server.strategy.FedAvg(),
)

# Start Flower client
fl.client.start_numpy_client("127.0.0.1:8080", client=CifarClient())
