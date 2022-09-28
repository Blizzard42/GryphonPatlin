import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten, Concatenate
import numpy as np
# import matplotlib.pyplot as plt

# Model Description
class MyModel(Model):
    def __init__(self, _thesis_model, _antithesis_model, _network_size):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.hidden = []
        if _antithesis_model is None: # If the anithesis is None, then we are looking for a standard network because it is not Hegelian
            for i in range(len(_network_size)):
                self.hidden.append([Dense(_network_size[i], activation="relu")])
        else:
            self.hidden = [[_thesis_model.hidden[0][0], _antithesis_model.hidden[0][0]], [Concatenate()]]
        self.outlayer = Dense(10)

    def call(self, x):
        x = self.flatten(x)
        for i in range(len(self.hidden)): # Iterate through each layer
            lx_arr = []
            for j in range(len(self.hidden[i])): # Iterate through each concatenation
                lx_arr.append(self.hidden[i][j](x))
            x = lx_arr if len(lx_arr) > 1 else lx_arr[0]
        return self.outlayer(x)


class NetworkUnit:
    # Model and Statistic Initialization
    def __init__(self, _train_ds, _test_ds, _sample_weight_multiplier, _sample_weight_adjuster, _thesis_model,
                 _antithesis_model, _network_size, _sample_weights):
        self.train_ds = _train_ds
        self.test_ds = _test_ds
        self.SAMPLE_WEIGHT_MULTIPLIER = _sample_weight_multiplier
        self.SAMPLE_WEIGHT_ADJUSTER = _sample_weight_adjuster
        self.model = MyModel(_thesis_model=_thesis_model, _antithesis_model=_antithesis_model, _network_size=_network_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_trainset_loss = tf.keras.metrics.Mean(name='test_trainset_loss')  # New
        self.test_trainset_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_trainset_accuracy')  # New

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.combined_test_accuracies = []

        self.sample_weights = _sample_weights
        self.output_weights = []
        self.score = None
        self.failures = None

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

    def train(self, epochs):
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
            self.combined_test_accuracies.append(self.test_accuracy.result().numpy())

        self.output_weights = sample_weight_out


    def calcScore(self):
        self.score = (self.output_weights == self.SAMPLE_WEIGHT_MULTIPLIER + self.SAMPLE_WEIGHT_ADJUSTER)
        self.failures = len(self.output_weights[self.score])
        return self.score, self.failures

    def compareErrors(self, comparison_score, comparison_errors):
        print(f'Initials Errors: {self.failures}')
        print(f'Competitors Errors: {comparison_errors}')
        print(f'Shared Errors: {len(self.output_weights[self.score & comparison_score])}')


    def print_data(self):
        for i in range(1, len(self.model.layers)):
            print('')
            print(f'Layer {self.model.layers[i]}: ')
            print('Weights: ', self.model.layers[i].weights)
            print('Biases: ', self.model.layers[i].bias.numpy())
            print('Bias Initializer: ', self.model.layers[i].bias_initializer)

def thesize(_train_ds, _test_ds, _network_size, _epochs):
    network = NetworkUnit(_train_ds=_train_ds, _test_ds=_test_ds, _network_size=_network_size, _thesis_model=None,
                          _antithesis_model=None, _sample_weights=None, _sample_weight_multiplier=1, _sample_weight_adjuster=0)
    network.train(epochs=_epochs)
    return network

def antithesize(_train_ds, _test_ds, _network_size, _epochs, _sample_weights):
    network = NetworkUnit(_train_ds=_train_ds, _test_ds=_test_ds, _network_size=_network_size, _thesis_model=None,
                          _antithesis_model=None, _sample_weights=_sample_weights, _sample_weight_multiplier=1,
                          _sample_weight_adjuster=0)
    network.train(epochs=_epochs)
    return network

def synthesize(_train_ds, _test_ds, _epochs, _thesis_model, _antithesis_model):
    network = NetworkUnit(_train_ds=_train_ds, _test_ds=_test_ds, _network_size=None, _thesis_model=_thesis_model,
                          _antithesis_model=_antithesis_model, _sample_weights=None, _sample_weight_multiplier=1,
                          _sample_weight_adjuster=0)
    network.train(epochs=_epochs)
    return network


    # thesis_score = thesis_sample_weights == sample_weight_multiplier + sample_weight_adjuster
    # antithesis_score = antithesis_sample_weights == sample_weight_multiplier + sample_weight_adjuster
    # aristotle_score = aristotle_sample_weights == sample_weight_multiplier + sample_weight_adjuster
    #
    # print(f'Number of Hegel Errors: {len(aristotle_score[aristotle_score])}')
    # print(f'Number of Thesis Shared Errors: {len(aristotle_score[thesis_score & aristotle_score])}')
    # print(f'Number of Antithesis Shared Errors: {len(aristotle_score[antithesis_score & aristotle_score])}')
    # print(f'Number of Blind Spots: {len(aristotle_score[thesis_score & aristotle_score & antithesis_score])}')
