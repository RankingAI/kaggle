from keras.models import Model

from keras.layers import Input, concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Dropout, Activation, Add, Dense, Multiply, Lambda
from keras.layers.merge import concatenate

from keras.optimizers import  Adam, SGD

from CyclicLearningRate import CyclicLR
from snapshot import SnapshotCallbackBuilder

import config
import metric_1
from lovasz_losses_tf import lovasz_hinge

import numpy as np
from tqdm import tqdm

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

class UNetWithResBlock:
    def __init__(self, input_shape= [None, None, 1], print_network=True, stages = 2):
        input_layer = Input(input_shape)
        output_layer = self.__get_network(input_layer,
                                          start_neurons= 16,
                                          dropout_ratio = 0.4,
                                          act= 'relu',
                                          stride_size= (1, 1),
                                          attention= False,
                                          dilated_bottleneck= False
                                          )

        self.stages = stages
        self.networks = []
        for i in range(self.stages):
            if(i == 0):
                inp = input_layer
                out = output_layer
            else:
                inp = self.networks[i - 1].layers[0].input
                out = self.networks[i - 1].layers[-1].input
            network = Model(inp, out)
            self.networks.append(network)

        if(print_network):
            for i in range(len(self.networks)):
                print('\n ----------------- Summary of Network %s ------------------' % i)
                self.networks[i].summary()

    def load_weight(self, weight_file, stage= -1):
        ''''''
        wf = '%s.%s' % (weight_file, stage)
        self.networks[stage].load_weights(wf)
        print('model file %s ' % wf)

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
            # early stopping
            early_stopping = EarlyStopping(monitor= monitor_s, mode='max', patience= 16, verbose=1)
            # model checkpoint
            model_checkpoint = ModelCheckpoint(model_weight_file, monitor= monitor_s,mode='max', save_best_only=True, verbose=1)
            # learning rate schedule
            lr_schedule = ReduceLROnPlateau(monitor= monitor_s, mode='max', factor=0.5, patience= 8, min_lr= 0.00001, verbose=1)

            callback_list = [model_checkpoint, early_stopping, lr_schedule]

        # compile
        net = self.networks[stage]
        net.compile(loss= loss_s, optimizer= opti, metrics=[metric_s])

        # fitting
        net.fit(X_train, Y_train, validation_data=[X_valid, Y_valid], epochs=epochs, batch_size=batch_size,callbacks=callback_list, verbose=2, shuffle= False)

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

    def __convolution_block(self, x, filters, size, strides=(1, 1), padding='same', act = None):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        if act:
            x = Activation(act)(x)
        return x

    def __residual_block(self, blockInput, num_filters=16, act= 'relu'):
        x = Activation(act)(blockInput)
        x = BatchNormalization()(x)
        x = self.__convolution_block(x, filters= num_filters, size= (3, 3), act= act)
        x = self.__convolution_block(x, filters= num_filters, size= (3, 3), act= None)
        x = Add()([x, blockInput])

        return x

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

    def __get_network(self, input_layer, start_neurons, dropout_ratio = 0.4, act= 'relu', stride_size= (2, 2), attention= True, dilated_bottleneck= False):

        print('Network params: ')
        print('start neural %s, dropout ratio %.2f, activate %s, stride %s, attention %s, dilated_bottleneck %s' % (start_neurons, dropout_ratio, act, stride_size, attention, dilated_bottleneck))

        # 128 -> 64
        conv1 = Conv2D(start_neurons * 1, kernel_size= (3, 3), strides= stride_size, dilation_rate= 1, padding="same")(input_layer)
        conv1 = self.__residual_block(conv1, start_neurons * 1, act= act)
        conv1 = self.__residual_block(conv1, start_neurons * 1, act= act)
        conv1 = self.__csse_block(conv1, 'conv1')

        conv1 = Activation(act)(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        # 64 -> 32
        conv2 = Conv2D(start_neurons * 2, kernel_size= (3, 3), strides= stride_size, dilation_rate= 1, padding="same")(pool1)
        conv2 = self.__residual_block(conv2, start_neurons * 2, act= act)
        conv2 = self.__residual_block(conv2, start_neurons * 2, act= act)
        conv2 = self.__csse_block(conv2, 'conv2')

        conv2 = Activation(act)(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        # 32 -> 16
        conv3 = Conv2D(start_neurons * 4, kernel_size= (3, 3), strides= stride_size, dilation_rate= 1, padding="same")(pool2)
        conv3 = self.__residual_block(conv3, start_neurons * 4, act= act)
        conv3 = self.__residual_block(conv3, start_neurons * 4, act= act)
        conv3 = self.__csse_block(conv3, 'conv3')

        conv3 = Activation(act)(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)

        # 16 -> 8
        conv4 = Conv2D(start_neurons * 8, kernel_size= (3, 3), strides= stride_size, dilation_rate= 1, padding="same")(pool3)
        conv4 = self.__residual_block(conv4, start_neurons * 8, act= act)
        conv4 = self.__residual_block(conv4, start_neurons * 8, act= act)
        conv4 = self.__csse_block(conv4, 'conv4')

        conv4 = Activation(act)(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)

        # Middle
        convm = Conv2D(start_neurons * 16, kernel_size= (3, 3), strides= stride_size, dilation_rate= 1, padding="same")(pool4)
        convm = self.__residual_block(convm, start_neurons * 16, act= act)
        convm = self.__residual_block(convm, start_neurons * 16, act= act)
        # dilated conv
        if(dilated_bottleneck):
            convm = self.__dilated_conv(convm, 3, start_neurons * 16)

        convm = Activation(act)(convm)

        # 8 -> 16
        deconv4 = UpSampling2D()(convm)
        if(attention):
            uconv4 = self.__attention_block(deconv4, conv4, start_neurons * 8)
        else:
            uconv4 = concatenate([deconv4, conv4])

        uconv4 = Conv2D(start_neurons * 8, kernel_size= (3, 3), padding="same")(uconv4)
        uconv4 = self.__residual_block(uconv4, start_neurons * 8, act= act)
        uconv4 = self.__residual_block(uconv4, start_neurons * 8, act= act)
        uconv4 = self.__csse_block(uconv4, 'uconv4')

        uconv4 = Activation(act)(uconv4)

        # 16 -> 32
        deconv3 = UpSampling2D()(uconv4)
        if(attention):
            uconv3 = self.__attention_block(deconv3, conv3, start_neurons * 4)
        else:
            uconv3 = concatenate([deconv3, conv3])

        uconv3 = Conv2D(start_neurons * 4, kernel_size= (3, 3), padding="same")(uconv3)
        uconv3 = self.__residual_block(uconv3, start_neurons * 4, act= act)
        uconv3 = self.__residual_block(uconv3, start_neurons * 4, act= act)
        uconv3 = self.__csse_block(uconv3, 'uconv3')

        uconv3 = Activation(act)(uconv3)

        # 32 -> 64
        deconv2 = UpSampling2D()(uconv3)
        if(attention):
            uconv2 = self.__attention_block(deconv2, conv2, start_neurons * 2)
        else:
            uconv2 = concatenate([deconv2, conv2])

        uconv2 = Conv2D(start_neurons * 2, kernel_size= (3, 3), padding="same")(uconv2)
        uconv2 = self.__residual_block(uconv2, start_neurons * 2, act= act)
        uconv2 = self.__residual_block(uconv2, start_neurons * 2, act= act)
        uconv2 = self.__csse_block(uconv2, 'uconv2')

        uconv2 = Activation(act)(uconv2)

        # 64 -> 128
        deconv1 = UpSampling2D()(uconv2)
        if(attention):
            uconv1 = self.__attention_block(deconv1, conv1, start_neurons * 1)
        else:
            uconv1 = concatenate([deconv1, conv1])

        uconv1 = Conv2D(start_neurons * 1, kernel_size= (3, 3), padding="same")(uconv1)
        uconv1 = self.__residual_block(uconv1, start_neurons * 1, act= act)
        uconv1 = self.__residual_block(uconv1, start_neurons * 1, act= act)
        uconv1 = self.__csse_block(uconv1, 'uconv1')

        uconv1 = Activation(act)(uconv1)

        ## hypercolumns
        hypercol = concatenate([uconv1, UpSampling2D(2)(uconv2), UpSampling2D(4)(uconv3), UpSampling2D(8)(uconv4)])
        hypercol = Conv2D(32, kernel_size= (3, 3), padding="same")(hypercol)

        final = hypercol

        final = Dropout(dropout_ratio)(final)
        output_layer_noact = Conv2D(1, kernel_size= (1, 1), padding="same")(final)
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

if __name__ == '__main__': # test network frame
    model = UNetWithResBlock(input_shape= [128, 128, 1], print_network= True)
