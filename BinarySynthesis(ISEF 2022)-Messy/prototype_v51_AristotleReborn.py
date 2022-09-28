import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten, Concatenate
import numpy as np
# import matplotlib.pyplot as plt

# Model Description
class MyModel(Model):
    def __init__(self, _thesis_model, _antithesis_model):
        super(MyModel, self).__init__()
        self.thesis_model = _thesis_model
        self.antithesis_model = _antithesis_model
        self.thesis_flattener = Flatten()
        self.antithesis_flattener = Flatten()
        self.flatten = Flatten()
        self.concat_input = Concatenate()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x2 = self.thesis_model(x, training=False)
        x2 = self.thesis_flattener(x2)
        x3 = self.antithesis_model(x, training=False)
        x3 = self.antithesis_flattener(x3)
        x = self.flatten(x)
        x = self.concat_input([x, x2, x3])
        x = self.d1(x)
        return self.d2(x)


class NetworkUnit:
    # Model and Statistic Initialization
    def __init__(self, _train_ds, _test_ds, _sample_weight_multiplier, _sample_weight_adjuster, _thesis_model,
                 _antithesis_model, _sample_weights=None):
        self.train_ds = _train_ds
        self.test_ds = _test_ds
        self.SAMPLE_WEIGHT_MULTIPLIER = _sample_weight_multiplier
        self.SAMPLE_WEIGHT_ADJUSTER = _sample_weight_adjuster
        self.model = MyModel(_thesis_model, _antithesis_model)

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

    def test_trainset_step(self, images, labels):
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
                for images, labels in self.train_ds:
                    self.train_step(images, labels, self.sample_weights[32*i:32*(i+1)])
                    i += 1
            else:
                for images, labels in self.train_ds:
                    self.train_step(images, labels)

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            for images, labels in self.train_ds:
                weighted = self.test_trainset_step(images, labels)
                sample_weight_out = np.append(sample_weight_out, (np.multiply(self.SAMPLE_WEIGHT_MULTIPLIER,
                                                                              weighted) + self.SAMPLE_WEIGHT_ADJUSTER))

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


def trial(_train_ds, _test_ds, sample_weight_multiplier, sample_weight_adjuster, _thesis_model, _antithesis_model,
          thesis_sample_weights, antithesis_sample_weights, thesis_epochs=1, antithesis_epochs=1, print_data=False):
    aristotle_network = NetworkUnit(_train_ds, _test_ds, sample_weight_multiplier, sample_weight_adjuster,
                                    _thesis_model, _antithesis_model)
    aristotle_sample_weights = aristotle_network.train()

    thesis_score = thesis_sample_weights == sample_weight_multiplier + sample_weight_adjuster
    antithesis_score = antithesis_sample_weights == sample_weight_multiplier + sample_weight_adjuster
    aristotle_score = aristotle_sample_weights == sample_weight_multiplier + sample_weight_adjuster

    print(f'Number of Aristotle Errors: {len(aristotle_score[aristotle_score])}')
    print(f'Number of Thesis Shared Errors: {len(aristotle_score[thesis_score & aristotle_score])}')
    print(f'Number of Antithesis Shared Errors: {len(aristotle_score[antithesis_score & aristotle_score])}')
    print(f'Number of Blind Spots: {len(aristotle_score[thesis_score & aristotle_score & antithesis_score])}')
    return aristotle_network.test_accuracy.result()
