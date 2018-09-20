import metric_1

import sys
import numpy as np
from keras.models import Model
import deeplab_v3

from CyclicLearningRate import CyclicLR
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Input, concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, SpatialDropout2D
from keras.layers import BatchNormalization, Dropout, Activation, Add
from keras.layers.merge import concatenate

from keras.utils.data_utils import get_file

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tqdm import tqdm

loss_metric_config = {
    0: [metric_1.bce_dice_loss, metric_1.my_iou_metric_0, 'val_my_iou_metric_0'],
    1: [metric_1.lovasz_loss, metric_1.my_iou_metric_1, 'val_my_iou_metric_1'],
}

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"

class DeeplabV3:
    ''''''
    def __init__(self, input_shape, freeze_till_layer= 'input_1', stages= 0, print_network= False, phase= 'train'):
        ''''''
        self.stages = stages
        self.input_shape = input_shape
        self.input_layer = Input(shape= input_shape)
        self.freeze_till_layer = freeze_till_layer

        self.output_layer = self.__get_network()

        self.networks = []

        # multiple stages
        for i in range(self.stages):
            if(i == 0):
                inp = self.input_layer
                out = self.output_layer
            else:
                inp = self.networks[i - 1].layers[0].input
                out = self.networks[i - 1].layers[-1].input
            network = Model(inp, out)
            self.networks.append(network)

        # if(print_network):
        #     for i in range(len(self.networks)):
        #         print('\n ----------------- Summary of Network %s ------------------' % i)
        #         self.networks[i].summary()
        #         break

        for no, l in enumerate(self.networks[0].layers):
            if(no >= 356):
                print('------------------')
            print(l.name)

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

        clr = CyclicLR(base_lr= 0.0001, max_lr= 0.001, step_size= 500)

        callback_list = [model_checkpoint, reduce_lr, early_stopping]

        # compile
        opti = Adam(lr= learning_rate)
        net = self.networks[stage]
        net.compile(loss= loss_s, optimizer= opti, metrics=[metric_s])

        # load weights
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',WEIGHTS_PATH_X,cache_subdir='models')
        net.load_weights(weights_path, by_name=True)

        # freeze
        self.__freeze_model(net, self.freeze_till_layer) # freeze few layers while training

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


    def __get_network(self):
        output_layer_noact = deeplab_v3.Deeplabv3(input_tensor= self.input_layer, input_shape= self.input_shape, classes= 1, backbone= 'xception')
        output_layer = Activation('sigmoid')(output_layer_noact)

        return output_layer

if __name__ == '__main__':
    DeeplabV3(input_shape= [128, 128, 3], stages= 1, print_network= True)
