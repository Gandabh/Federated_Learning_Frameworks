import os
import flwr as fl
import tensorflow as tf

# Set TensorFlow to log less
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

# Create the model
model = create_custom_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

# Define the Flower client class
class CifarClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # Adjust data for each client using the client_id
        client_x_train = x_train[self.client_id * len(x_train) // 3 : (self.client_id + 1) * len(x_train) // 3]
        client_y_train = y_train[self.client_id * len(y_train) // 3 : (self.client_id + 1) * len(y_train) // 3]
        model.fit(client_x_train, client_y_train, epochs=1, batch_size=32)
        # Log information
        print(f"Client {self.client_id}: Fit complete")
        return model.get_weights(), len(client_x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # Adjust data for each client using the client_id
        client_x_test = x_test[self.client_id * len(x_test) // 3 : (self.client_id + 1) * len(x_test) // 3]
        client_y_test = y_test[self.client_id * len(y_test) // 3 : (self.client_id + 1) * len(y_test) // 3]
        loss, accuracy = model.evaluate(client_x_test, client_y_test)
        # Log information
        print(f"Client {self.client_id}: Evaluation complete - Loss: {loss}, Accuracy: {accuracy}")
        return loss, len(client_x_test), {"accuracy": accuracy}

if __name__ == "__main__":
    # Start Flower clients for 3 nodes
    for client_id in range(3):
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient(client_id))
