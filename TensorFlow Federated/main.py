import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split the dataset into clients
def create_tf_dataset_for_client(x, y, num_clients=10):
    client_data = []
    for i in range(num_clients):
        start = i * len(x) // num_clients
        end = (i + 1) * len(x) // num_clients
        client_data.append(tf.data.Dataset.from_tensor_slices((x[start:end], y[start:end]))
                           .map(lambda img, lbl: (tf.image.resize(img, (32, 32)), lbl))
                           .repeat(10).batch(20))
    return client_data

# Create client datasets
num_clients = 10
train_data = create_tf_dataset_for_client(x_train, y_train, num_clients)

# Define a Keras model for use with TFF
def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return tff.learning.models.from_keras_model(
        model,
        input_spec=train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Simulate a few rounds of training with the selected client devices
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001)
)
state = trainer.initialize()

for round_num in range(5):
    state, metrics = trainer.next(state, train_data)
    print(f'Round {round_num+1}, Metrics={metrics}')
