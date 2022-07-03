from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import src.python.yalefaces as yalefaces

def __optimizer__(learn_rate, momentum, optimizer):
    if optimizer == "SGD":
        optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    elif optimizer == "Adam":
        optimizer = Adam(learning_rate=learn_rate)
    elif optimizer == "Adadelta":
        optimizer = Adadelta(learning_rate=learn_rate)
    elif optimizer == "Adamax":
        optimizer = Adamax(learning_rate=learn_rate)
    elif optimizer == "RMSprop":
        optimizer = RMSprop(learning_rate=learn_rate, momentum=momentum)
    return optimizer


def CNN(optimizer='adam', learn_rate=1, momentum=0):
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

    optimizer = __optimizer__(learn_rate, momentum, optimizer)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def Lenet5():
    # optimizer = compile_kwargs.get('optimizer', 'adam')
    # learn_rate = compile_kwargs.get('learn_rate', 1)
    # momentum = compile_kwargs.get('momentum', 0)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(28, 5, activation="tanh", strides=1, padding="same", input_shape=[243, 320, 1]))
    model.add(tf.keras.layers.AveragePooling2D((14, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(10, 5, activation="tanh", strides=1, padding="same"))
    model.add(tf.keras.layers.AveragePooling2D((5, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(1, 5, activation="tanh", strides=1, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(15, activation="softmax"))

    # optimizer = __optimizer__(learn_rate, momentum, optimizer)
    # model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import src.python.yalefaces as yalefaces

    X, y = yalefaces.load("../../datasets/yalefaces", flatten=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11, stratify=y)

    model = Lenet5()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_test, y_test))

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
