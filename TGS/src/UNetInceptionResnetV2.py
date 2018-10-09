import metric_1

import sys
import numpy as np

from inception_resnet_v2 import InceptionResNetV2

import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import UpSampling2D, Conv2D, SpatialDropout2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, concatenate, Input, Dropout, Dense, Lambda, Multiply, Add

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tqdm import tqdm

from lovasz_losses_tf import lovasz_hinge
import config
from snapshot import SnapshotCallbackBuilder

loss_metric_config = {
    0: [metric_1.bce_dice_loss, metric_1.my_iou_metric_0, 'val_my_iou_metric_0'],
    1: [metric_1.lovasz_loss, metric_1.my_iou_metric_1, 'val_my_iou_metric_1'],
    2: [metric_1.lovasz_loss, metric_1.my_iou_metric_1, 'val_my_iou_metric_1'],
}

# activation, elu + 1
def elupone(x, alpha= 1.0):
    return K.elu(x, alpha) + 1.0

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

        self.output_layer = self.__get_network(start_neurons= 8, dropout_ratio= 0.4, act= 'relu', attention= False, dilated_bottleneck= False)

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
            elif (i == 1):
                inp = self.networks[i - 1].layers[0].input
                out = self.networks[i - 1].layers[-1].input
            else:
                inp = self.networks[i - 1].layers[0].input
                out = self.networks[i - 1].layers[-1].output
            network = Model(inp, out)
            self.networks.append(network)

        if(print_network):
            for i in range(len(self.networks)):
                print('\n ----------------- Summary of Network %s ------------------' % i)
                self.networks[i].summary()
                break

    def load_weight(self, weight_file, stage= -1):
        ''''''
        self.networks[stage].load_weights(weight_file)
        print('load model %s done.' % weight_file)

    def fit(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, model_weight_file, learning_rate, stage=0, snapshot_ensemble= True):

        loss_s = loss_metric_config[stage][0]
        metric_s = loss_metric_config[stage][1]
        monitor_s = loss_metric_config[stage][2]

        # callbacks
        if(snapshot_ensemble):
            opti = SGD()
            callback_list = SnapshotCallbackBuilder(nb_epochs= epochs, nb_snapshots= 3, init_lr= learning_rate, monitor= monitor_s, model_weight_prefix= model_weight_file).get_callbacks()
        else:
            opti = Adam(lr= learning_rate)
            # model checkpoint
            model_checkpoint = ModelCheckpoint(model_weight_file, monitor= monitor_s,mode='max', save_best_only=True, verbose=1)
            if(stage == 0):
                # early stopping
                early_stopping = EarlyStopping(monitor= monitor_s, mode='max', patience= 20, verbose=1)
                # learning rate schedule
                lr_schedule = ReduceLROnPlateau(monitor= monitor_s, mode='max', factor=0.5, patience= 10, min_lr= 0.00001, verbose=1)
            else:
                # early stopping
                early_stopping = EarlyStopping(monitor= monitor_s, mode='max', patience= 8, verbose=1)
                # learning rate schedule
                lr_schedule = ReduceLROnPlateau(monitor= monitor_s, mode='max', factor=0.5, patience= 4, min_lr= 0.00001, verbose=1)

            callback_list = [model_checkpoint, early_stopping, lr_schedule]

        # compile
        net = self.networks[stage]
        net.compile(loss= loss_s, optimizer= opti, metrics=[metric_s])

        # fitting
        net.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], epochs=epochs, batch_size=batch_size,callbacks=callback_list, verbose=2, shuffle= True)

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
        if(loss_metric_config[stage][2] == 'val_my_iou_metric_1'):
            print('using logit thresholds...')
            thresholds = np.array([np.log(v / (1.0 - v)) for v in thresholds])  # transform into logits for the last model
        elif(loss_metric_config[stage][2] == 'val_my_iou_metric_0'):
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

    def __cse_block(self, prevlayer, prefix):
        mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
        lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
        lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
        x = Multiply()([prevlayer, lin2])
        return x

    def __sse_block(self, prevlayer, prefix):
        conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
                      activation='sigmoid', strides=(1, 1),
                      name=prefix + "_conv")(prevlayer)
        conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
        return conv

    def __csse_block(self, x, prefix):
        '''
        Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
        https://arxiv.org/abs/1803.02579
        '''
        cse = self.__cse_block(x, prefix)
        sse = self.__sse_block(x, prefix)
        x = Add(name=prefix + "_csse_mul")([cse, sse])

        return x

    def __attention_block(self, x, shortcut, i_filters):
        g1 = Conv2D(i_filters, kernel_size=1)(shortcut)
        g1 = BatchNormalization()(g1)
        x1 = Conv2D(i_filters, kernel_size=1)(x)
        x1 = BatchNormalization()(x1)

        g1_x1 = Add()([g1, x1])
        psi = Activation('relu')(g1_x1)
        psi = Conv2D(1, kernel_size=1)(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        x_ = Multiply()([x, psi])

        return x_

    def __get_network(self, start_neurons= 8, dropout_ratio = 0.4, act= 'relu', stride_size= (2, 2), attention= False, dilated_bottleneck= False):

        # 256 -> 128
        conv1 = self.base_model.model.layers[self.encoder_layers['activation_3']].output
        conv1 = self.__csse_block(conv1, 'conv1')

        # 128 -> 64
        conv2 = self.base_model.model.layers[self.encoder_layers['activation_5']].output
        conv2 = self.__csse_block(conv2, 'conv2')

        # 64 -> 32
        conv3 = self.base_model.model.layers[self.encoder_layers['block35_10_ac']].output
        conv3 = self.__csse_block(conv3, 'conv3')

        # 32 -> 16
        conv4 = self.base_model.model.layers[self.encoder_layers['block17_20_ac']].output
        conv4 = self.__csse_block(conv4, 'conv4')

        # center, 16 -> 8
        conv5 = self.base_model.model.layers[self.encoder_layers['conv_7b_ac']].output
        # dilated conv
        if(dilated_bottleneck):
            conv5 = self.__dilated_conv(conv5, 3, start_neurons * 128)
            conv5 = Activation(act)(conv5)

        # 8 -> 16
        if(attention):
            conv6 = self.__attention_block(UpSampling2D()(conv5), conv4, start_neurons * 64)
        else:
            up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
            conv6 = self.__conv_block_simple(up6, start_neurons * 64, "conv6_1")
            conv6 = self.__conv_block_simple(conv6, start_neurons * 64, "conv6_2")

        conv6 = self.__csse_block(conv6, 'conv6')

        # 16 -> 32
        if(attention):
            conv7 = self.__attention_block(UpSampling2D()(conv6), conv3, start_neurons * 32)
        else:
            up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
            conv7 = self.__conv_block_simple(up7, start_neurons * 32, "conv7_1")
            conv7 = self.__conv_block_simple(conv7, start_neurons * 32, "conv7_2")

        conv7 = self.__csse_block(conv7, 'conv7')

        # 32 -> 64
        if(attention):
            conv8 = self.__attention_block(UpSampling2D()(conv7), conv2, start_neurons * 16)
        else:
            up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
            conv8 = self.__conv_block_simple(up8, start_neurons * 16, "conv8_1")
            conv8 = self.__conv_block_simple(conv8, start_neurons * 16, "conv8_2")

        conv8 = self.__csse_block(conv8, 'conv8')

        # 64 -> 128
        if(attention):
            conv9 = self.__attention_block(UpSampling2D()(conv8), conv1, start_neurons * 8)
        else:
            up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
            conv9 = self.__conv_block_simple(up9, start_neurons * 8, "conv9_1")
            conv9 = self.__conv_block_simple(conv9, start_neurons * 8, "conv9_2")

        conv9 = self.__csse_block(conv9, 'conv9')

        # 128 -> 256
        if(attention):
            conv10 = self.__attention_block(UpSampling2D()(conv9), self.base_model.model.input, start_neurons * 4)
        else:
            up10 = concatenate([UpSampling2D()(conv9), self.base_model.model.input], axis=-1)
            conv10 = self.__conv_block_simple(up10, start_neurons * 4, "conv10_1")
            conv10 = self.__conv_block_simple(conv10, start_neurons * 4, "conv10_2")

        conv10 = self.__csse_block(conv10, 'conv10')

        # hypercolumns
        hypercol = concatenate([conv10,
                                UpSampling2D(size= (2, 2))(conv9),
                                UpSampling2D(size= (4, 4))(conv8),
                                UpSampling2D(size= (8, 8))(conv7),
                                #UpSampling2D(size= (16, 16))(conv6),
                                ])
        #hypercol = self.__conv_block_simple(hypercol, start_neurons * 4, "hypercol")
        hypercol = Conv2D(start_neurons * 4, kernel_size= (3, 3), padding="same", name= 'hypercol')(hypercol)

        final = SpatialDropout2D(dropout_ratio)(hypercol)
        #final = Dropout(0.2)(conv10)

        output_layer_noact = Conv2D(1, (1, 1), name="out")(final)
        output_layer = Activation('sigmoid')(output_layer_noact)

        return output_layer

    def __dilated_conv(self, x, depth, filters):
        ''''''
        # dilated conv
        dilated_layers = []
        for i in range(depth):
            x = Conv2D(filters, [3, 3], activation= 'relu', padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return Add()(dilated_layers)

if __name__ == '__main__':
    model = UNetWithResNet(input_shape= [256, 256, 3], print_network= True, stages= 1)

