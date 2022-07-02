from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# reference: https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
class LeNet:
	@staticmethod
	def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
		
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

        # define the first set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(24, 3, padding="same", input_shape=inputShape))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(48, 5, padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(64, 7, padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(800))
		model.add(Activation(activation))
		
        # define the second FC layer
		model.add(Dense(numClasses))
		
        # lastly, define the soft-max classifier
		model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)
		
		return model