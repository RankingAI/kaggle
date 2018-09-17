import metric_1

import sys
import numpy as np
import Xception
from keras.models import Model

import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, SpatialDropout2D
from keras.layers import BatchNormalization, Dropout, Activation, Add
from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tqdm import tqdm

loss_metric_config = {
    1: [metric_1.bce_dice_loss, metric_1.my_iou_metric_0, 'val_my_iou_metric_0'],
    0: [metric_1.lovasz_loss, metric_1.my_iou_metric_1, 'val_my_iou_metric_1'],
}

class UNetXception:
    ''''''
    def __init__(self, input_shape, freeze_till_layer= 'input_1', stages= 0, print_network= False, phase= 'train'):
        ''''''
        self.stages = stages
        self.input_layer = Input(shape= input_shape)
        self.base_model = Xception.Xception(include_top= False, input_shape= input_shape, input_tensor= self.input_layer)
        for l in self.base_model.model.layers:
            l.trainable = True
        #self.base_model.summary()
        self.encoder_layers = {'block1_conv1_act': 3, 'block1_conv2_act': 6, 'block3_sepconv1_act': 16, 'block4_sepconv1_act': 26, 'block13_sepconv1_act': 116, 'block14_sepconv2_act': 131}

        self.output_layer = self.__get_network()
        #### run with it out of training
        # for layer_name in self.encoder_layers.keys():
        #     for i, l in enumerate(self.base_model.layers):
        #         if(layer_name == l.name):
        #             self.encoder_layers[layer_name] = i
        #             break
        print(self.encoder_layers)

        self.networks = []

        # multiple stages
        for i in range(self.stages):
            if(i == 0):
                inp = self.base_model.model.input
                out = self.output_layer
            else:
                inp = self.networks[i - 1].layers[0].input
                out = self.networks[i - 1].layers[-1].input
            network = Model(inp, out)
            self.networks.append(network)

        if(print_network):
            for i in range(len(self.networks)):
                print('\n ----------------- Summary of Network %s ------------------' % i)
                self.networks[i].summary()
                break

    def load_weight(self, weight_file, stage= -1):
        ''''''
        wf = '%s.%s' % (weight_file, stage)
        self.networks[stage].load_weights(wf)
        print('model file %s ' % wf)

    def fit(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, model_weight_file, learning_rate, stage=0):

        loss_s = loss_metric_config[stage][0]
        metric_s = loss_metric_config[stage][1]
        monitor_s = loss_metric_config[stage][2]

        # early stopping
        early_stopping = EarlyStopping(monitor= monitor_s, mode='max', patience= 10, verbose=1)

        # save the best checkpoint
        model_checkpoint = ModelCheckpoint('%s.%s' % (model_weight_file, stage), monitor= monitor_s,mode='max', save_best_only=True, verbose=1)

        # dynamic reduce the learning rate
        reduce_lr = ReduceLROnPlateau(monitor= monitor_s, mode='max', factor=0.5, patience=5, min_lr= 0.00001, verbose=1)

        callback_list = [model_checkpoint, reduce_lr, early_stopping]

        # compile
        opti = Adam(lr= learning_rate)
        net = self.networks[stage]
        # self.__freeze_model(net, self.freeze_till_layer) # freeze few layers while training
        net.compile(loss= loss_s, optimizer= opti, metrics=[metric_s])

        # fitting
        net.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], epochs=epochs, batch_size=batch_size,callbacks=callback_list, verbose=2)

    def predict(self, X_test, stage= -1):
        ''''''
        preds_test_1 = self.networks[stage].predict(X_test)

        # predict on the flipped one
        x_test_reflect = np.array([np.fliplr(x) for x in X_test])
        preds_test_refect = self.networks[stage].predict(x_test_reflect)
        preds_test_2 = np.array([np.fliplr(x) for x in preds_test_refect])

        # average the tuple
        preds_avg = (preds_test_1 + preds_test_2) / 2

        return preds_avg

    def evaluate(self, Pred_valid, Y_valid, stage= -1):
        ''''''
        print('shape of predicted')
        print(Pred_valid.shape)
        print('shap of truth')
        print(Y_valid.shape)

        thresholds = np.linspace(0.1, 0.9, 75)
        if ((stage != 0) & (stage == self.stages - 1)):
            print('using logit thresholds...')
            thresholds = np.array([np.log(v / (1.0 - v)) for v in thresholds])  # transform into logits for the last model
        else:
            print('using proba thresholds...')

        # iou at different thresholds
        ious = np.array([metric_1.iou_metric_batch(Y_valid, np.int32(Pred_valid > threshold)) for threshold in tqdm(thresholds)])

        # the best threshold
        threshold_best_index = np.argmax(ious)
        threshold_best = thresholds[threshold_best_index]

        # the best iou
        iou_best = ious[threshold_best_index]

        return iou_best, threshold_best

    # def __conv_block_simple(self, prevlayer, filters, prefix, strides=(1, 1)):
    #     conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides,
    #                   name=prefix + "_conv")(prevlayer)
    #     conv = BatchNormalization(name=prefix + "_bn")(conv)
    #     conv = Activation('relu', name=prefix + "_activation")(conv)
    #     return conv

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

    def __get_network(self):
        ''''''
        conv0 = self.base_model.model.layers[self.encoder_layers['block1_conv1_act']].output
        conv1 = self.base_model.model.layers[self.encoder_layers['block1_conv2_act']].output
        conv2 = self.base_model.model.layers[self.encoder_layers['block3_sepconv1_act']].output
        conv3 = self.base_model.model.layers[self.encoder_layers['block4_sepconv1_act']].output
        conv4 = self.base_model.model.layers[self.encoder_layers['block13_sepconv1_act']].output
        conv5 = self.base_model.model.layers[self.encoder_layers['block14_sepconv2_act']].output

        # 3 -> 6
        deconv5 = Conv2DTranspose(2048, (3, 3), strides= (2, 2), padding="same")(conv5)
        uconv4  = concatenate([deconv5, conv4]) # 2048 + 728
        uconv4 = Dropout(0.5)(uconv4)

        uconv4 = Conv2D(1024, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = self.__residual_block(uconv4, 1024)
        uconv4 = self.__residual_block(uconv4, 1024)
        uconv4 = Activation('relu')(uconv4)

        uconv4 = Conv2D(512, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = self.__residual_block(uconv4, 512)
        uconv4 = self.__residual_block(uconv4, 512)
        uconv4 = Activation('relu')(uconv4)

        # 6 -> 12
        deconv4 = Conv2DTranspose(256, (3, 3), strides= (2, 2), padding= 'same')(uconv4)
        uconv3  = concatenate([deconv4, conv3]) # 256 + 256
        uconv3 = Dropout(0.5)(uconv3)

        uconv3 = Conv2D(256, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = self.__residual_block(uconv3, 256)
        uconv3 = self.__residual_block(uconv3, 256)
        uconv3 = Activation('relu')(uconv3)

        # 12 -> 24
        deconv3 = Conv2DTranspose(128, (3, 3), strides= (2, 2), padding= 'same')(uconv3)
        uconv2  = concatenate([deconv3, conv2]) # 128 + 128
        uconv2 = Dropout(0.5)(uconv2)

        uconv2 = Conv2D(128, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = self.__residual_block(uconv2, 128)
        uconv2 = self.__residual_block(uconv2, 128)
        uconv2 = Activation('relu')(uconv2)

       # 24 -> 48
        deconv2 = Conv2DTranspose(64, (3, 3), strides= (2, 2), padding= 'same')(uconv2)
        uconv1  = concatenate([deconv2, conv1]) # 64 + 64
        uconv1 = Dropout(0.5)(uconv1)

        uconv1 = Conv2D(64, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = self.__residual_block(uconv1, 64)
        uconv1 = self.__residual_block(uconv1, 64)
        uconv1 = Activation('relu')(uconv1)

        # 48 -> 50
        deconv1 = Conv2DTranspose(32, (3, 3), strides= (1, 1), padding= 'valid')(uconv1)
        uconv0  = concatenate([deconv1, conv0]) # 64 + 64
        uconv0 = Dropout(0.5)(uconv0)

        uconv0 = Conv2D(32, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = self.__residual_block(uconv0, 32)
        uconv0 = self.__residual_block(uconv0, 32)
        uconv0 = Activation('relu')(uconv0)

        # 50 -> 101
        deconv0 = Conv2DTranspose(16, (3, 3), strides= (2, 2), padding= 'valid')(uconv0)
        uinput  = concatenate([deconv0, self.base_model.model.input]) # 32 + 32
        uinput = Dropout(0.5)(uinput)

        uinput = Conv2D(16, (3, 3), activation=None, padding="same")(uinput)
        uinput = self.__residual_block(uinput, 16)
        uinput = self.__residual_block(uinput, 16)
        uinput = Activation('relu')(uinput)

        uinput = Dropout(0.25)(uinput)
        output_layer_noact = Conv2D(1, (1, 1), name="out")(uinput)
        output_layer = Activation('sigmoid')(output_layer_noact)

        return output_layer_noact

if __name__ == '__main__':
    UNetXception(input_shape= [101, 101, 3])

