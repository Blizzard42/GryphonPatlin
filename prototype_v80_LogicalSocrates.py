import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten
import numpy as np


# Model Description
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_nums = [784, 128, 10]
        self.flatten = Flatten()
        self.d1 = Dense(self.layer_nums[1], activation='relu')
        self.d2 = Dense(self.layer_nums[2])

    def call(self, x, layer_array=[1, 1, 1], node_array=[0, 0, 0]):
        x = self.flatten(x)
        x = (x * layer_array[0]) + node_array[0]
        x = self.d1(x)
        x = (x * layer_array[1]) + node_array[1]
        x = self.d2(x)
        x = (x * layer_array[2]) + node_array[2]
        return x

    def change_weight(self, coordinate=[0, 0], new_weight=1):
        print(self.weights)


class NetworkUnit:
    # Model and Statistic Initialization
    def __init__(self, _train_ds, _test_ds, _sample_weights=None):
        self.train_ds = _train_ds
        self.test_ds = _test_ds
        self.model = MyModel()
        self.score = []
        self.accuracy = 0.0

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_trainset_loss = tf.keras.metrics.Mean(name='test_trainset_loss')  # New
        self.test_trainset_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_trainset_accuracy')  # New

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

    def test_trainset_step(self, images, labels):  # New
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_trainset_loss(t_loss)
        self.test_trainset_accuracy(labels, predictions)

        output = np.argmax(predictions, axis=1)
        output = (output != labels)
        return output

    def test_node_step(self, images, labels, layer_array, node_array):
        predictions = self.model(x=images, layer_array=layer_array, node_array=node_array, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_trainset_loss(t_loss)
        self.test_trainset_accuracy(labels, predictions)

        output = np.argmax(predictions, axis=1)
        output = (output != labels)
        return output

    def change_weight(self, coordinate, ):
        print("test")

    def test(self):
        score_summary = [0, 0, 0]
        for i in range(len(self.model.layer_nums)):  #
            for j in range(self.model.layer_nums[i]):  #
                score = np.ndarray(shape=(32, 0))
                node_array = [np.zeros(self.model.layer_nums[0]),
                              np.zeros(self.model.layer_nums[1]), np.zeros(self.model.layer_nums[2])]
                node_array[i][j] = 1
                layer_array = [np.ones(self.model.layer_nums[0]),
                               np.ones(self.model.layer_nums[1]), np.ones(self.model.layer_nums[2])]
                layer_array[i][j] = -1
                self.test_trainset_loss.reset_states()
                self.test_trainset_accuracy.reset_states()
                for images, labels in self.train_ds:
                    weighted = self.test_node_step(images=images, labels=labels, layer_array=layer_array,
                                                   node_array=node_array)
                    score = np.append(score, weighted)
                score_summary[0] += len(score[(score == 1) & (self.score == 1)]) / \
                                    len(score[score == 1])
                score_summary[1] += 1
                score_summary[2] = max(score_summary[2], self.test_trainset_accuracy.result() * 100 - self.accuracy)
                print(
                    f'Direct Accuracy: {self.test_trainset_accuracy.result() * 100 - self.accuracy}'
                )
            print("---------------------")
        score_summary[0] = score_summary[0] / score_summary[1]
        return score_summary

    def train(self, epochs=1):
        # Implementation
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_trainset_loss.reset_states()
            self.test_trainset_accuracy.reset_states()
            samples = np.ndarray(shape=(32, 0))

            if self.sample_weights is not None:
                i = 0
                for images, labels in self.train_ds:
                    self.train_step(images, labels, self.sample_weights[32 * i:32 * (i + 1)])
                    i += 1
            else:
                for images, labels in self.train_ds:
                    self.train_step(images, labels)

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            for images, labels in self.train_ds:
                weighted = self.test_trainset_step(images, labels)
                samples = np.append(samples, weighted)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result() * 100}, '
                f'Test Loss: {self.test_loss.result()}, '
                f'Test Accuracy: {self.test_accuracy.result() * 100}, '
                f'Direct Accuracy: {self.test_trainset_accuracy.result() * 100}'
            )

        self.score = samples
        self.accuracy = self.test_trainset_accuracy.result() * 100
        return samples

    def print_data(self):
        for i in range(1, len(self.model.layers)):
            print('')
            print(f'Layer {self.model.layers[i]}: ')
            print('Weights: ', self.model.layers[i].weights)
            print('Biases: ', self.model.layers[i].bias.numpy())
            print('Bias Initializer: ', self.model.layers[i].bias_initializer)


def trial(_train_ds, _test_ds, epochs=1, _sample_weights=None, print_data=False):
    thesis_model = NetworkUnit(_train_ds=_train_ds, _test_ds=_test_ds, _sample_weights=_sample_weights)
    answers = thesis_model.train(epochs=epochs)
    score_summary = thesis_model.test()

    number_of_failures_T = len(answers[answers == 1])
    print('Number of Failures (Thesis): ', number_of_failures_T)

    average_crossover = score_summary[0]
    print('Average Crossover: ', average_crossover)

    max_accuracy = score_summary[2]
    print('Max Accuracy: ', max_accuracy)

    return thesis_model.model, answers, thesis_model.test_accuracy.result()
    # number_of_failures_T, number_of_failures_AT, number_of_shared_failures
