import metric_1

import sys
import numpy as np

from inception_resnet_v2 import InceptionResNetV2

import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import UpSampling2D, Conv2D, SpatialDropout2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, concatenate, Input

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tqdm import tqdm

from lovasz_losses_tf import lovasz_hinge
import config

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    # logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss

class UNetWithResNet:
    ''''''
    def __init__(self, input_shape= [None, None, 3], freeze_till_layer= 'input_1', print_network= False, stages= 0, phase= 'train'):
        ''''''
        self.stages = stages
        self.freeze_till_layer = freeze_till_layer
        self.encoder_layers = {'activation_3': 9, 'activation_5': 16, 'block35_10_ac': 260, 'block17_20_ac': 594, 'conv_7b_ac': 779}

        self.input_layer = Input(shape= input_shape)
        self.base_model = InceptionResNetV2(include_top= False, input_tensor= self.input_layer, input_shape= input_shape, phase= phase)

        self.output_layer = self.__get_network()

        print('\n -------- encoder layers ---------')
        print(self.encoder_layers)

        #### run with it out of training
        # for layer_name in self.encoder_layers.keys():
        #     for i, l in enumerate(base_model.layers):
        #         if(layer_name == l.name):
        #             self.encoder_layers[layer_name] = i
        #             break

        self.networks = []

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

    #def reset(self, stage):
    #    self.networks[stage] = None

    #def __del__(self):
    #    self.base_model = None
    #    for s in range(self.stages):
    #        self.networks[s] = None

    def load_weight(self, weight_file, stage= -1):
        ''''''
        wf = '%s.%s' % (weight_file, stage)
        self.networks[stage].load_weights(wf)
        print('model file %s ' % wf)

    def fit(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, model_weight_file, learning_rate, stage=0):

        # early stopping
        early_stopping = EarlyStopping(monitor='val_my_iou_metric_%s' % stage, mode='max', patience= 10, verbose=1)

        # save the best checkpoint
        model_checkpoint = ModelCheckpoint('%s.%s' % (model_weight_file, stage), monitor='val_my_iou_metric_%s' % stage,mode='max', save_best_only=True, verbose=1)
                                           
        # dynamic reduce the learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_%s' % stage, mode='max', factor=0.5, patience=5,min_lr=0.00001, verbose=1)

        callback_list = []
        callback_list.append(model_checkpoint)
        callback_list.append(reduce_lr)
        if (stage == self.stages - 1):
            print('adding early stopping mechanism...')
            callback_list.append(early_stopping)

        # compile
        opti = Adam(lr= learning_rate)
        net = self.networks[stage]
        #self.__freeze_model(net, self.freeze_till_layer) # freeze few layers while training
        if(stage == 0):
            net.compile(loss= metric_1.bce_dice_loss, optimizer = opti, metrics= [metric_1.my_iou_metric_0])
        elif(stage == 1):
            net.compile(loss= lovasz_loss, optimizer = opti, metrics= [metric_1.my_iou_metric_1])

        # fitting
        net.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], epochs=epochs,batch_size=batch_size, callbacks=callback_list, verbose=2, shuffle= False)

    def predict(self, X_test, stage= -1):
        ''''''
        preds_test_1 = self.networks[stage].predict(X_test)
        #print('prediction on original image done.')

        # predict on the flipped one
        x_test_reflect = np.array([np.fliplr(x) for x in X_test])
        preds_test_refect = self.networks[stage].predict(x_test_reflect)
        preds_test_2 = np.array([np.fliplr(x) for x in preds_test_refect])
        #print('pridiction on the flipped one done.')

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

    def __freeze_model(self, model, freeze_before_layer):
        if freeze_before_layer == "ALL":
            for l in model.layers:
                l.trainable = False
        else:
            freeze_before_layer_index = -1
            for i, l in enumerate(model.layers):
                if l.name == freeze_before_layer:
                    freeze_before_layer_index = i
            for l in model.layers[:freeze_before_layer_index]:
                l.trainable = False

    def __conv_block_simple(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides,
                      name=prefix + "_conv")(prevlayer)
        conv = BatchNormalization(name=prefix + "_bn")(conv)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv

    def __get_network(self):

        conv1 = self.base_model.model.layers[self.encoder_layers['activation_3']].output
        conv2 = self.base_model.model.layers[self.encoder_layers['activation_5']].output
        conv3 = self.base_model.model.layers[self.encoder_layers['block35_10_ac']].output
        conv4 = self.base_model.model.layers[self.encoder_layers['block17_20_ac']].output
        conv5 = self.base_model.model.layers[self.encoder_layers['conv_7b_ac']].output

        #up6 = concatenate([Conv2DTranspose(256, (3, 3), strides= (2,2), padding= 'same')(conv5), conv5])
        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = self.__conv_block_simple(up6, 256, "conv6_1")
        conv6 = self.__conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = self.__conv_block_simple(up7, 256, "conv7_1")
        conv7 = self.__conv_block_simple(conv7, 256, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = self.__conv_block_simple(up8, 128, "conv8_1")
        conv8 = self.__conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = self.__conv_block_simple(up9, 64, "conv9_1")
        conv9 = self.__conv_block_simple(conv9, 64, "conv9_2")

        up10 = concatenate([UpSampling2D()(conv9), self.base_model.model.input], axis=-1)
        conv10 = self.__conv_block_simple(up10, 48, "conv10_1")
        conv10 = self.__conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(0.4)(conv10)

        output_layer_noact = Conv2D(1, (1, 1), name="out")(conv10)
        output_layer = Activation('sigmoid')(output_layer_noact)

        return output_layer

if __name__ == '__main__':
    model = UNetWithResNet(input_shape= [256, 256, 3], print_network= True, stages= 1)

