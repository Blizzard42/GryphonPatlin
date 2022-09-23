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
    def __init__(self, _thesis_model, _antithesis_model):
        super(MyModel, self).__init__()
        self.thesis_model = _thesis_model
        self.antithesis_model = _antithesis_model
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.concat_input = Concatenate()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x2 = self.thesis_model(x, training=False)
        x2 = self.flatten2(x2)
        x3 = self.antithesis_model(x, training=False)
        x3 = self.flatten3(x3)
        x = self.flatten1(x)
        x = self.concat_input([x, x2, x3])
        x = self.d1(x)
        return self.d2(x)


class NetworkUnit:
    # Model and Statistic Initialization
    def __init__(self, thesis, antithesis, _sample_weights=None):
        self.model = MyModel(thesis, antithesis)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_trainset_loss = tf.keras.metrics.Mean(name='test_trainset_loss')  # New
        self.test_trainset_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_trainset_accuracy')  # New

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.sample_weights = _sample_weights

    # Train and Test Descriptions
    @tf.function
    def train_step(self, images, labels, sample_weights=None):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions, sample_weight=sample_weights)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def test_trainset_step(self, images, labels): # New
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_trainset_loss(t_loss)
        self.test_trainset_accuracy(labels, predictions)
        output = np.argmax(predictions, axis=1)
        output = (output != labels)
        return output

    def train(self, epochs=1):
        # Implementation
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_trainset_loss.reset_states()
            self.test_trainset_accuracy.reset_states()
            sample_weight_out = np.ndarray(shape=(32, 0))

            if self.sample_weights is not None:
                i = 0
                for images, labels in train_ds:
                    self.train_step(images, labels, self.sample_weights[32*i:32*(i+1)])
                    i += 1
            else:
                for images, labels in train_ds:
                    self.train_step(images, labels)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels)

            for images, labels in train_ds:
                weighted = self.test_trainset_step(images, labels)
                sample_weight_out = np.append(sample_weight_out, np.multiply(SAMPLE_WEIGHT, weighted) + 1)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result() * 100}, '
                f'Test Loss: {self.test_loss.result()}, '
                f'Test Accuracy: {self.test_accuracy.result() * 100}, '
                f'Direct Accuracy: {self.test_trainset_accuracy.result() * 100}'
            )

        return sample_weight_out

    def print_data(self):
        for i in range(1, len(self.model.layers)):
            print('')
            print(f'Layer {self.model.layers[i]}: ')
            print('Weights: ', self.model.layers[i].weights)
            print('Biases: ', self.model.layers[i].bias.numpy())
            print('Bias Initializer: ', self.model.layers[i].bias_initializer)


def trial(thesis_network, antithesis_network, t_sample_weights, at_sample_weights, epochs=1, print_data=False):
    aristotle = NetworkUnit(thesis_network, antithesis_network)
    sample_weights = aristotle.train(epochs)
    thesis_shared_failures = len(
        sample_weights[((sample_weights == SAMPLE_WEIGHT + 1) & (t_sample_weights == SAMPLE_WEIGHT + 1))])
    antithesis_shared_failures = len(
        sample_weights[((sample_weights == SAMPLE_WEIGHT + 1) & (at_sample_weights == SAMPLE_WEIGHT + 1))])
    t_at_shared_failures = ((t_sample_weights == SAMPLE_WEIGHT + 1) & (at_sample_weights == SAMPLE_WEIGHT + 1))
    blind_spots = len(
        sample_weights[((sample_weights == SAMPLE_WEIGHT + 1) & t_at_shared_failures)])
    print('Number of Shared Failures (Thesis): ', thesis_shared_failures)
    print('Number of Shared Failures (Antithesis): ', antithesis_shared_failures)
    print('Blind Spots: ', blind_spots)
    return aristotle.test_accuracy.result()
