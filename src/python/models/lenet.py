from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD

# reference: https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
class LeNet:
	@staticmethod
	def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None, num_filters=[24, 32], filter_sizes=[5, 3], dense_width=256):
		# K.clear_session()

		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# define the first set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(num_filters[0], filter_sizes[0], padding="same", input_shape=inputShape, activation=activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		for i in range(1, len(num_filters)):
			# define the first set of CONV => ACTIVATION => POOL layers
			model.add(Conv2D(num_filters[i], filter_sizes[i], padding="same", activation=activation))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
        # define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(dense_width, activation=activation))
		
        # define the second FC layer
		model.add(Dense(numClasses, activation='softmax'))

        # if a weights path is supplied (inicating that the model was pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)

		model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
		
		return model