import metric_1

import numpy as np

# pre-trained models
from resnet50_fixed import ResNet50
from keras.applications.vgg16 import VGG16

import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import UpSampling2D, Conv2D, SpatialDropout2D, Conv2DTranspose, MaxPooling2D
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

class UNetResNet50VGG16:
    def __init__(self, input_shape= [None, None, 3], freeze_till_layer= 'input_1', learning_rate= 0.001, print_network= False, stages= 1):
        ''''''
        self.stages = stages

        resnet_model = ResNet50(input_shape=input_shape, include_top=False)

        vgg_model = VGG16(input_shape=input_shape, input_tensor= resnet_model.input, include_top=False)

        output_layer = self.__get_network(resnet_model, vgg_model)

        self.networks = []
        opti = Adam(lr= learning_rate)
        ## model 0
        network_0 = Model(resnet_model.input, output_layer)
        self.__freeze_model(network_0, freeze_till_layer) # freeze few layers while training

        network_0.compile(loss= metric_1.bce_dice_loss, optimizer= opti,metrics= [metric_1.my_iou_metric_0])
        self.networks.append(network_0)

        if(print_network):
            for i in range(len(self.networks)):
                print('\n ----------------- Summary of Network %s ------------------' % i)
                self.networks[i].summary()

    # NOT used yet
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

    def load_weight(self, weight_file, stage=1):
        ''''''
        wf = '%s.%s' % (weight_file, stage)
        self.networks[stage].load_weights(wf)
        print('model file %s ' % wf)

    def fit(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, model_weight_file, stage=0):
        # early stopping
        early_stopping = EarlyStopping(monitor='val_my_iou_metric_%s' % stage, mode='max', patience=20, verbose=1)

        # save the best checkpoint
        model_checkpoint = ModelCheckpoint('%s.%s' % (model_weight_file, stage), monitor='val_my_iou_metric_%s' % stage,mode='max', save_best_only=True, verbose=1)

        # dynamic reduce the learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_%s' % stage, mode='max', factor=0.5, patience=5,min_lr=0.00001, verbose=1)

        callback_list = []
        callback_list.append(model_checkpoint)
        callback_list.append(reduce_lr)
        if (stage == self.stages - 1): # add early stopping controller, the last stage
            callback_list.append(early_stopping)
        self.networks[stage].fit(X_train, Y_train, validation_data=[X_valid, Y_valid], epochs=epochs,batch_size=batch_size, callbacks=callback_list, verbose=2)

    def predict(self, X_test, stage=0):
        ''''''
        preds_test_1 = self.networks[stage].predict(X_test)

        # predict on the flipped one
        x_test_reflect = np.array([np.fliplr(x) for x in X_test])
        preds_test_refect = self.networks[stage].predict(x_test_reflect)
        preds_test_2 = np.array([np.fliplr(x) for x in preds_test_refect])

        # average the tuple
        preds_avg = (preds_test_1 + preds_test_2) / 2

        return preds_avg

    def evaluate(self, Pred_valid, Y_valid, stage= 0):
        ''''''
        thresholds = np.linspace(0.1, 0.9, 75)
        if (stage == self.stages - 1):
            thresholds = np.array([np.log(v / (1.0 - v)) for v in thresholds])  # transform into logits for the last model

        # iou at different thresholds
        ious = np.array([metric_1.iou_metric_batch(Y_valid, np.int32(Pred_valid > threshold)) for threshold in tqdm(thresholds)])

        # the best threshold
        threshold_best_index = np.argmax(ious)
        threshold_best = thresholds[threshold_best_index]

        # the best iou
        iou_best = ious[threshold_best_index]

        return iou_best, threshold_best

    def __conv_block_simple(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides,
                      name=prefix + "_conv")(prevlayer)
        conv = BatchNormalization(name=prefix + "_bn")(conv)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv

    def __get_network(self, resnet_model, vgg_model):

        for l in resnet_model.layers:
            l.trainable = True
        conv1 = resnet_model.get_layer("activation_1").output
        conv2 = resnet_model.get_layer("activation_10").output
        conv3 = resnet_model.get_layer("activation_22").output
        conv4 = resnet_model.get_layer("activation_40").output
        conv5 = resnet_model.get_layer("activation_49").output

        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = self.__conv_block_simple(up6, 256, "conv6_1")
        conv6 = self.__conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = self.__conv_block_simple(up7, 192, "conv7_1")
        conv7 = self.__conv_block_simple(conv7, 192, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = self.__conv_block_simple(up8, 128, "conv8_1")
        conv8 = self.__conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = self.__conv_block_simple(up9, 64, "conv9_1")
        conv9 = self.__conv_block_simple(conv9, 64, "conv9_2")

        for l in vgg_model.layers:
            l.trainable = False
        vgg_first_conv = vgg_model.get_layer("block1_conv2").output
        up10 = concatenate([UpSampling2D()(conv9), resnet_model.input, vgg_first_conv], axis=-1)
        conv10 = self.__conv_block_simple(up10, 32, "conv10_1")
        conv10 = self.__conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(0.2)(conv10)
        output_layer_noact = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
        output_layer = Activation('sigmoid')(output_layer_noact)

        return output_layer

if __name__ == '__main__':
    model = UNetResNet50VGG16(input_shape= [256, 256, 3], print_network= True)