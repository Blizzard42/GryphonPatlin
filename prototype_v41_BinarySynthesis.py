import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten, Concatenate
import numpy as np


# Data acquisition and Handling
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
CUTOFF = int(len(x_train)/1)  # Change the integer to adjust the portion of data cut
SAMPLE_WEIGHT = 1
x_train = x_train[:CUTOFF].copy()
y_train = y_train[:CUTOFF].copy()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Model Description
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten_former_predictions = Flatten()  # New!
        # self.num_feeder_networks = len(_former_predictions)
        # self.feeder_flatteners = []
        # self.feeder_networks_predictions = _former_predictions
        # for network in _former_networks:
        #     self.feeder_flatteners.append(Flatten())
        self.flatten = Flatten()
        self.concat_input = Concatenate()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x, x2):
        # inputs = []
        # for i in range(self.num_feeder_networks):
        #     value = self.feeder_networks[i](x, training=False)
        #     inputs.append(self.feeder_flatteners[i](value))
        x = self.flatten(x)
        x2 = self.flatten_former_predictions(x2)  # New!
        # inputs.append(x)
        x = self.concat_input([x, x2])
        x = self.d1(x)
        return self.d2(x)


class NetworkUnit:
    # Model and Statistic Initialization
    def __init__(self, _former_predictions=np.zeros([1, 60000, 10]), _sample_weights=[]):
        self.model = MyModel()
        self.former_predictions = np.transpose(_former_predictions, (1, 0, 2))

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_trainset_loss = tf.keras.metrics.Mean(name='test_trainset_loss')
        self.test_trainset_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_trainset_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.sample_weights = _sample_weights

    # Train and Test Descriptions
    @tf.function
    def train_step(self, images, labels, _former_predictions, _sample_weights):
        with tf.GradientTape() as tape:
            predictions = self.model(images, _former_predictions, training=True)
            loss = self.loss_object(labels, predictions, sample_weight=_sample_weights)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, np.zeros(()), training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def test_trainset_step(self, images, labels, _former_predictions):
        predictions = self.model(images, _former_predictions, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_trainset_loss(t_loss)
        self.test_trainset_accuracy(labels, predictions)
        score = np.argmax(predictions, axis=1)
        score = (score != labels)
        score = np.multiply(SAMPLE_WEIGHT, score) + 1
        return predictions, score

    def train(self, epochs=1):
        # Implementation
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_trainset_loss.reset_states()
            self.test_trainset_accuracy.reset_states()
            score_whole = np.ndarray(shape=(32, 0))
            predictions_whole = np.ndarray(shape=(32, 0))

            i = 0
            for images, labels in train_ds:
                self.train_step(images=images,
                                labels=labels,
                                _former_predictions=self.former_predictions[i:i+32],
                                _sample_weights=self.sample_weights[i: i+32] if len(self.sample_weights) > 0
                                else None)
                i += 32

            for test_images, test_labels in test_ds:
                self.test_step(images=test_images,
                               labels=test_labels)

            i = 0
            for images, labels in train_ds:
                predictions, score = self.test_trainset_step(images=images,
                                                             labels=labels,
                                                             _former_predictions=self.former_predictions[i:i+32])
                predictions_whole = np.concatenate((predictions_whole, predictions[np.newaxis, :, :]), axis=0)
                score_whole = np.append(score_whole, score)
                i += 32

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result() * 100}, '
                f'Test Loss: {self.test_loss.result()}, '
                f'Test Accuracy: {self.test_accuracy.result() * 100}, '
                f'Direct Accuracy: {self.test_trainset_accuracy.result() * 100}'
            )

        return predictions_whole, score_whole

    def print_data(self):
        for i in range(1, len(self.model.layers)):
            print('')
            print(f'Layer {self.model.layers[i]}: ')
            print('Weights: ', self.model.layers[i].weights)
            print('Biases: ', self.model.layers[i].bias.numpy())
            print('Bias Initializer: ', self.model.layers[i].bias_initializer)


def trial(former_predictions, blind_spots=np.array([True]), epochs=1, print_data=False):
    binary_synthesis = NetworkUnit(former_predictions)
    predictions, score = binary_synthesis.train(epochs)

    former_predictions = np.concatenate((former_predictions, predictions[np.newaxis, :, :]), axis=0)
    print(former_predictions)

    errors = (score == SAMPLE_WEIGHT + 1)
    blind_spots = (blind_spots & errors)

    print(f'Blind Spots: {len(blind_spots[blind_spots])}')

    return former_predictions, blind_spots
