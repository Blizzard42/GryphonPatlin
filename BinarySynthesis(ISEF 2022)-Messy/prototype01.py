import tensorflow as tf
import keras
import numpy as np

def neuralNet():
    # Data Import
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Data Processing
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Establish Model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1)
    orig_loss, orig_acc = model.evaluate(train_images, train_labels, verbose=2)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(model.metrics)

    return test_acc, orig_acc
