import tensorflow as tf
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay

import src.python.yalefaces as yalefaces


class CNN:

    @staticmethod
    def build_model(activation='relu'):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, 12, activation=activation, padding="same", input_shape=[243, 320, 1]))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Conv2D(128, 3, activation=activation, padding="same"))
        model.add(tf.keras.layers.Conv2D(128, 3, activation=activation, padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Conv2D(256, 3, activation=activation, padding="same"))
        model.add(tf.keras.layers.Conv2D(256, 3, activation=activation, padding="same"))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=activation))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation=activation))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(15, activation="softmax"))
        return model

    @staticmethod
    def run(X_train, y_train, epochs=10, n_iter=10, fold_size=5):
        batch_size = [20, 40, 60, 80]
        optimizer = ['sgd', 'rmsprop', 'adadelta', 'adam', 'adamx']
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        activation = ['relu', 'tanh']
        momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
        #              'he_uniform']
        # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid = dict(
            batch_size=batch_size,
            epochs=[epochs],
            optimizer=optimizer,
            optimizer__learning_rate=learn_rate,
            model__activation=activation
        )
        model = KerasClassifier(model=CNN.build_model, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        cv = RandomizedSearchCV(model, param_grid, n_jobs=-1, cv=StratifiedKFold(n_splits=fold_size, shuffle=True),
                                n_iter=n_iter)
        grid_result = cv.fit(X_train, y_train)
        return grid_result, cv


class Lenet5:

    @staticmethod
    def build_model():
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(28, 5, activation="tanh", strides=1, padding="same", input_shape=[243, 320, 1]))
        model.add(tf.keras.layers.AveragePooling2D((14, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(10, 5, activation="tanh", strides=1, padding="same"))
        model.add(tf.keras.layers.AveragePooling2D((5, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(1, 5, activation="tanh", strides=1, padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(15, activation="softmax"))
        return model

    @staticmethod
    def run(X_train, y_train, epochs=10, n_iter=10, fold_size=5):
        batch_size = [20, 40, 60, 80]

        optimizer = ['sgd', 'rmsprop', 'adadelta', 'adam', 'adamx']
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
        #              'he_uniform']
        # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid = dict(
            batch_size=batch_size,
            epochs=[epochs],
            optimizer=optimizer,
            optimizer__learning_rate=learn_rate
        )

        model = KerasClassifier(model=Lenet5.build_model, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        cv = RandomizedSearchCV(model, param_grid, n_jobs=-1, cv=StratifiedKFold(n_splits=fold_size, shuffle=True),
                                n_iter=n_iter)
        # cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=4)
        grid_result = cv.fit(X_train, y_train)
        return grid_result, cv

    def fit(self, X, y, params={}):
        self.model = KerasClassifier(model=Lenet5.build_model, loss="sparse_categorical_crossentropy",
                                     metrics=['accuracy'], **params)
        self.model.fit(X_train, y_train)

    def loss_display(self):
        history = self.model.model_.history_
        plt.plot(history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def accuray_display(self):
        history = self.model.model_.history_
        plt.plot(history['accuracy'], label='accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

    def evaluate(self, X_test, y_test):
        test_loss, test_acc = cv.best_estimator_.model_.evaluate(X_test, y_test)
        print("Score (original):", test_acc)
        y_pred = self.model.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred.argmax(axis=1))


def cv():
    batch_size = [20, 40, 60, 80]
    epochs = [20]
    optimizer = ['sgd', 'rmsprop', 'adadelta', 'adam', 'adamx']
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
    #              'he_uniform']
    # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        optimizer__learning_rate=learn_rate
    )
    model = KerasClassifier(model=Lenet5, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    cv = RandomizedSearchCV(model, param_grid, n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True), n_iter=20)
    # cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=4)
    grid_result = cv.fit(X_train, y_train)
    return


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
