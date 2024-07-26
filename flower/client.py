import os
import flwr as fl
import tensorflow as tf
import logging
import sys
import time
import psutil
import numpy as np

# Set TensorFlow to log less
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a custom CNN model suitable for CIFAR-10
def create_custom_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

# Define the Flower client class
class CifarClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_clients):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = create_custom_model()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Split the data for this client
        idx_start = self.client_id * len(x_train) // self.num_clients
        idx_end = (self.client_id + 1) * len(x_train) // self.num_clients
        self.client_x_train = x_train[idx_start:idx_end]
        self.client_y_train = y_train[idx_start:idx_end]
        idx_start = self.client_id * len(x_test) // self.num_clients
        idx_end = (self.client_id + 1) * len(x_test) // self.num_clients
        self.client_x_eval = x_test[idx_start:idx_end]
        self.client_y_eval = y_test[idx_start:idx_end]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Train for 15 local epochs
        history = self.model.fit(self.client_x_train, self.client_y_train, epochs=15, batch_size=32, verbose=0)
        
        # Log CPU and RAM usage
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        logger.info(f"Client {self.client_id}: CPU usage: {cpu_usage}%, RAM usage: {ram_usage}%")
        
        return self.model.get_weights(), len(self.client_x_train), {"loss": history.history["loss"][-1], "accuracy": history.history["accuracy"][-1]}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.client_x_eval, self.client_y_eval, verbose=0)
        return loss, len(self.client_x_eval), {"accuracy": accuracy}

def client_fn(cid: str) -> fl.client.Client:
    # Parse client ID and create a client instance
    client_id = int(cid)
    return CifarClient(client_id=client_id, num_clients=NUM_CLIENTS)

if __name__ == "__main__":
    NUM_CLIENTS = 100  #client size changed here 1, 10, 50, or 100 as needed
    NUM_ROUNDS = 10

    # Start the client
    start_time = time.time()
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client_fn(sys.argv[1]))
    end_time = time.time()

    # Log total training time
    total_time = end_time - start_time
    logger.info(f"Total training time: {total_time:.2f} seconds")

    #Network utilization measured externally using tools like iftop
