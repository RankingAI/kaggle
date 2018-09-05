from keras.models import Model

from keras.layers import Input, concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Dropout, Activation, Add
from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import utils
import config
import metric_1
from tqdm import tqdm

class UNetWithResBlock:
	def __init__(self,print_network= True):
		input_layer = Input((config.img_size_original, config.img_size_original, 1))
		output_layer = self.__get_network(input_layer, 16, 0.5)

		self.network = Model(input_layer, output_layer)
		self.network.compile(loss="binary_crossentropy", optimizer="adam", metrics=[metric_1.my_iou_metric])

		if(print_network):
			self.network.summary()

	def load_weight(self, weight_file):
		''''''
		self.network.load_weights(weight_file)

	def fit(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, model_weight_file):
		early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode='max', patience=20, verbose=1)
		model_checkpoint = ModelCheckpoint(model_weight_file, monitor='val_my_iou_metric',
										   mode='max', save_best_only=True, verbose=1)
		reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.2, patience=5, min_lr=0.00001,
									  verbose=1)

		history = self.network.fit(X_train, Y_train,
							validation_data=[X_valid, Y_valid],
							epochs= epochs,
							batch_size= batch_size,
							callbacks=[early_stopping, model_checkpoint, reduce_lr],
							verbose=2)

	def predict(self, X_test):
		''''''
		x_test_reflect = np.array([np.fliplr(x) for x in X_test])
		preds_test_1 = self.network.predict(X_test).reshape(-1, config.img_size_original, config.img_size_original)
		preds_test_refect = self.network.predict(x_test_reflect).reshape(-1, config.img_size_original, config.img_size_original)
		preds_test_2 = np.array([np.fliplr(x) for x in preds_test_refect])
		preds_avg = (preds_test_1 + preds_test_2) / 2

		return preds_avg

	def evaluate(self, Pred_valid, Y_valid):
		''''''
		thresholds = np.linspace(0.3, 0.7, 31)
		ious = np.array([metric_1.iou_metric_batch(Y_valid, np.int32(Pred_valid > threshold)) for threshold in tqdm(thresholds)])

		threshold_best_index = np.argmax(ious)
		iou_best = ious[threshold_best_index]
		threshold_best = thresholds[threshold_best_index]

		return iou_best, threshold_best

	def __convolution_block(self, x, filters, size, strides=(1, 1), padding='same', activation=True):
		x = Conv2D(filters, size, strides=strides, padding=padding)(x)
		x = BatchNormalization()(x)
		if activation == True:
			x = Activation('relu')(x)
		return x

	def __residual_block(self, blockInput, num_filters=16):
		x = Activation('relu')(blockInput)
		x = BatchNormalization()(x)
		x = self.__convolution_block(x, num_filters, (3, 3))
		x = self.__convolution_block(x, num_filters, (3, 3), activation=False)
		x = Add()([x, blockInput])

		return x

	def __get_network(self, input_layer, start_neurons, DropoutRatio = 0.5):
		# 101 -> 50
		conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
		conv1 = self.__residual_block(conv1, start_neurons * 1)
		conv1 = self.__residual_block(conv1, start_neurons * 1)
		conv1 = Activation('relu')(conv1)
		pool1 = MaxPooling2D((2, 2))(conv1)
		pool1 = Dropout(DropoutRatio / 2)(pool1)

		# 50 -> 25
		conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
		conv2 = self.__residual_block(conv2, start_neurons * 2)
		conv2 = self.__residual_block(conv2, start_neurons * 2)
		conv2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D((2, 2))(conv2)
		pool2 = Dropout(DropoutRatio)(pool2)

		# 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = self.__residual_block(conv3, start_neurons * 4)
		conv3 = self.__residual_block(conv3, start_neurons * 4)
		conv3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D((2, 2))(conv3)
		pool3 = Dropout(DropoutRatio)(pool3)

		# 12 -> 6
		conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
		conv4 = self.__residual_block(conv4, start_neurons * 8)
		conv4 = self.__residual_block(conv4, start_neurons * 8)
		conv4 = Activation('relu')(conv4)
		pool4 = MaxPooling2D((2, 2))(conv4)
		pool4 = Dropout(DropoutRatio)(pool4)

		# Middle
		convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
		convm = self.__residual_block(convm, start_neurons * 16)
		convm = self.__residual_block(convm, start_neurons * 16)
		convm = Activation('relu')(convm)

		# 6 -> 12
		deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv4 = concatenate([deconv4, conv4])
		uconv4 = Dropout(DropoutRatio)(uconv4)

		uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
		uconv4 = self.__residual_block(uconv4, start_neurons * 8)
		uconv4 = self.__residual_block(uconv4, start_neurons * 8)

		uconv4 = Activation('relu')(uconv4)

		# 12 -> 25
		# deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
		uconv3 = concatenate([deconv3, conv3])
		uconv3 = Dropout(DropoutRatio)(uconv3)

		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = self.__residual_block(uconv3, start_neurons * 4)
		uconv3 = self.__residual_block(uconv3, start_neurons * 4)
		uconv3 = Activation('relu')(uconv3)

		# 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, conv2])

		uconv2 = Dropout(DropoutRatio)(uconv2)
		uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = self.__residual_block(uconv2, start_neurons * 2)
		uconv2 = self.__residual_block(uconv2, start_neurons * 2)
		uconv2 = Activation('relu')(uconv2)

		# 50 -> 101
		# deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
		uconv1 = concatenate([deconv1, conv1])

		uconv1 = Dropout(DropoutRatio)(uconv1)
		uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = self.__residual_block(uconv1, start_neurons * 1)
		uconv1 = self.__residual_block(uconv1, start_neurons * 1)
		uconv1 = Activation('relu')(uconv1)

		uconv1 = Dropout(DropoutRatio / 2)(uconv1)
		output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

		return output_layer
