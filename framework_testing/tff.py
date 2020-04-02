from comet_ml import Experiment
import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
import tensorflow_federated as tff
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds

exp = Experiment(workspace="federated-learning", project_name="compare_frameworks")
exp.set_name("tff_cache_prefetch_non_iid")


# Load simulation data.
# source, _ = tff.simulation.datasets.emnist.load_data()


# def client_data(n):
#     trf_record = lambda e: (tf.expand_dims(e["image"], -1), e["label"])
#     return (
#         source.create_tf_dataset_for_client(source.client_ids[n])
#         .map(trf_record)
#         .repeat(5)
#         .batch(50)
#     )


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


# Pick a subset of client devices to participate in training.
train_data, test_data = tfds.load("mnist", split=["train", "test"], as_supervised=True)
train_data_0 = train_data.filter(lambda x, y: y < 5)
train_data_1 = train_data.filter(lambda x, y: y >= 5)
train_data = [train_data_0, train_data_1]
train_data = [
    td.map(normalize_img).cache().repeat(5).batch(50).prefetch(tf.data.experimental.AUTOTUNE) for td in train_data
]

# Grab a single batch of data so that TFF knows what data looks like.
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(train_data[0]).next())


# Wrap a Keras model for use with TFF.
def model_fn():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    return tff.learning.from_keras_model(
        model,
        dummy_batch=sample_batch,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
    model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01)
)
state = trainer.initialize()
for i in range(50):
    state, metrics = trainer.next(state, train_data)
    exp.log_metric("loss", metrics.loss, i)
    exp.log_metric("acc", metrics.sparse_categorical_accuracy, i)
    if metrics.sparse_categorical_accuracy > 0.99:
        break
