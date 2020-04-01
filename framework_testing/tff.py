from comet_ml import Experiment
import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
import tensorflow_federated as tff
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

exp = Experiment(workspace="federated-learning", project_name="compare_frameworks")
exp.set_name("tff")


# Load simulation data.
source, _ = tff.simulation.datasets.emnist.load_data()


def client_data(n):
    trf_record = lambda e: (tf.expand_dims(e["pixels"], -1), e["label"])
    return (
        source.create_tf_dataset_for_client(source.client_ids[n])
        .map(trf_record)
        .repeat(5)
        .batch(50)
    )


# Pick a subset of client devices to participate in training.
train_data = [client_data(n) for n in range(2)]

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
    model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1)
)
state = trainer.initialize()
for _ in range(50):
    state, metrics = trainer.next(state, train_data)
    exp.log_metric("loss", metrics.loss)
    exp.log_metric("acc", metrics.sparse_categorical_accuracy)
