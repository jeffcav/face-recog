import tensorflow as tf


class CNN():

    def __init__(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, 12, activation="relu", padding="same", input_shape=[243, 320, 1]))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"))
        model.add(tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"))
        model.add(tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(15, activation="softmax"))

        # Compile the model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model

    def fit(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(X_train, y_train, epochs=30,
                                 validation_data=(X_test, y_test))
        return history

class Lenet5():

    def __init__(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(28, 5, activation="tanh", strides=1, padding="same", input_shape=[243, 320, 1]))
        model.add(tf.keras.layers.AveragePooling2D((14, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(10, 5, activation="tanh", strides=1, padding="same"))
        model.add(tf.keras.layers.AveragePooling2D((5, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(1, 5, activation="tanh", strides=1, padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(15, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model

    def fit(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(X_train, y_train, epochs=30,
                                 validation_data=(X_test, y_test))
        return history