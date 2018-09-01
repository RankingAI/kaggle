from keras.models import Model

from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import utils
import config
import metric
from tqdm import tqdm

class UNetModel():
	''''''
	def __init__(self, img_shape,
		 out_ch= 1,
		 start_ch= 64,
		 depth= 4,
		 inc_rate= 2.,
		 activation= 'relu',
		 dropout= 0.5,
		 batch_norm= False,
		 max_pooling= True,
		 up_sampling= True,
		 residual= False,
		 print_network= True):
		''''''
		input_net = Input(shape=img_shape)
		output_net = self.level_block(input_net,
								 start_ch,
								 depth,
								 inc_rate,
								 activation,
								 dropout,
								 batch_norm,
								 max_pooling,
							 	up_sampling,
								residual)
		output_net = Conv2D(out_ch, 1, activation='sigmoid')(output_net)
		self.network = Model(inputs= input_net, outputs= output_net)
		self.network.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
		if(print_network):
			self.network.summary()

	def load_weight(self, weight_file):
		''''''
		self.network.load_weights(weight_file)

	def fit(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, model_weight_file):
		''''''
		early_stopping = EarlyStopping(patience=10, verbose=1)
		model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, verbose=1)
		reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose= -1)

		history = self.network.fit(X_train, Y_train,
								   validation_data= [X_valid, Y_valid],
								   epochs= epochs,batch_size= batch_size,
								   callbacks= [early_stopping, model_checkpoint, reduce_lr],
								   shuffle=True, verbose= 20)

	def predict(self, X_test):
		''''''
		return self.network.predict(X_test).reshape(-1, config.img_size_target, config.img_size_target)

	def evaluate(self, Pred_valid, Y_valid):
		''''''
		Pred_valid = np.array([utils.downsample(x) for x in Pred_valid])

		thresholds = np.linspace(0, 1, 50)
		ious = np.array([metric.iou_metric_batch(Y_valid, np.int32(Pred_valid > threshold)) for threshold in tqdm(thresholds)])

		threshold_best_index = np.argmax(ious[9:-10]) + 9
		iou_best = ious[threshold_best_index]
		threshold_best = thresholds[threshold_best_index]

		return iou_best, threshold_best

	# conv block within u-net framework
	def conv_block(self, input_net, conv_dim, activate, batch_norm, skip_connect, dropout= 0.0):
		''''''
		output_net = Conv2D(conv_dim, 3, activation= activate, padding='same')(input_net)
		output_net = BatchNormalization()(output_net) if batch_norm else output_net
		output_net = Dropout(dropout)(output_net) if dropout else output_net
		output_net = Conv2D(conv_dim, 3, activation= activate, padding='same')(output_net)
		output_net = BatchNormalization()(output_net) if batch_norm else output_net

		return Concatenate()([input_net, output_net]) if skip_connect else output_net

	# recursive level block within u-net framework
	def level_block(self, input_net, conv_dim, depth, inc, activate, dropout, batch_norm, max_pooling, up_sampling, skip_conncet):
		''''''
		if depth > 0:
			n = self.conv_block(input_net, conv_dim, activate, batch_norm, skip_conncet)
			m = MaxPooling2D()(n) if max_pooling else Conv2D(conv_dim, 3, strides= 2, padding='same')(n)
			m = self.level_block(m, int(inc * conv_dim), depth - 1, inc, activate, dropout, batch_norm, max_pooling, up_sampling, skip_conncet)
			if up_sampling:
				m = UpSampling2D()(m)
				m = Conv2D(conv_dim, 2, activation= activate, padding='same')(m)
			else:
				m = Conv2DTranspose(conv_dim, 3, strides= 2, activation= activate, padding='same')(m)
			n = Concatenate()([n, m])
			m = self.conv_block(n, conv_dim, activate, batch_norm, skip_conncet)
		else:
			m = self.conv_block(input_net, conv_dim, activate, batch_norm, skip_conncet, dropout)

		return m
