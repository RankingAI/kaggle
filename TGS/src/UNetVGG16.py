import metric_1

import numpy as np

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

class UNetVGG16:
    def __init__(self, input_shape= [None, None, 3], freeze_till_layer= 'input_1', learning_rate= 0.001, print_network= False, stages= 1):
        ''''''
        self.stages = stages

        input_layer = Input(shape= input_shape)
        base_model = VGG16(include_top= False, input_tensor= input_layer)

        output_layer = self.__get_network(base_model)

        self.networks = []

        opti = Adam(lr= learning_rate)
        ## model 0
        network_0 = Model(input_layer, output_layer)
        #self.__freeze_model(network_0, freeze_till_layer) # freeze few layers while training

        network_0.compile(loss= metric_1.bce_dice_loss, optimizer= opti,metrics= [metric_1.my_iou_metric_0])
        self.networks.append(network_0)

        # # model 1, NOT usable so far
        # network_1 = Model(input_layer, network_0.layers[-1].input)
        # #self.__freeze_model(network_1, freeze_till_layer) # freeze few layers while training
        # network_1.compile(loss= lovasz_loss, optimizer= opti, metrics= [metric_1.my_iou_metric_1])
        # self.networks.append(network_1)

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
        if (stage == 1):
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

    def __get_network(self, base_model):
        # freeze all
        for l in base_model.layers:
            l.trainable = True
        conv1 = base_model.get_layer("block1_conv2").output
        conv2 = base_model.get_layer("block2_conv2").output
        conv3 = base_model.get_layer("block3_conv3").output
        pool3 = base_model.get_layer("block3_pool").output

        conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block4_conv1")(pool3)
        conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block4_conv2")(conv4)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

        conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block5_conv1")(pool4)
        conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block5_conv2")(conv5)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

        conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block6_conv1")(pool5)
        conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block6_conv2")(conv6)
        pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(conv6)

        conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block7_conv1")(pool6)
        conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal",
                       name="block7_conv2")(conv7)

        up8 = concatenate([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal",
                                           strides=(2, 2), padding='same')(conv7), conv6], axis=3)
        conv8 = Conv2D(384, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up8)

        up9 = concatenate([Conv2DTranspose(256, (3, 3), activation="relu", kernel_initializer="he_normal",
                                           strides=(2, 2), padding='same')(conv8), conv5], axis=3)
        conv9 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up9)

        up10 = concatenate([Conv2DTranspose(192, (3, 3), activation="relu", kernel_initializer="he_normal",
                                            strides=(2, 2), padding='same')(conv9), conv4], axis=3)
        conv10 = Conv2D(192, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up10)

        up11 = concatenate([Conv2DTranspose(128, (3, 3), activation="relu", kernel_initializer="he_normal",
                                            strides=(2, 2), padding='same')(conv10), conv3], axis=3)
        conv11 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up11)

        up12 = concatenate([Conv2DTranspose(64, (3, 3), activation="relu", kernel_initializer="he_normal",
                                            strides=(2, 2), padding='same')(conv11), conv2], axis=3)
        conv12 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up12)

        up13 = concatenate([Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal",
                                            strides=(2, 2), padding='same')(conv12), conv1], axis=3)
        conv13 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up13)

        output_layer_noact = Conv2D(1, (1, 1), name='out')(conv13)
        output_layer = Activation("sigmoid")(output_layer_noact)

        return output_layer

if __name__ == '__main__':
    model = UNetVGG16(input_shape= [128, 128, 3], print_network= True)