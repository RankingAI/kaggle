#################################
# End-2-End pipeline for models #
#################################
import os, sys, time, gc
import numpy as np
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm

from skimage.io import imread, imsave

import tensorflow as tf
import keras.backend as K

import config
import data_utils
import utils
import UNet

# configuration for GPU resources
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

input_shape = (config.img_size_target,config.img_size_target,1)

def train(train_data, ModelWeightDir, EvaluateFile):
    ''''''
    # CV
    cv_iou = np.zeros(config.kfold)
    cv_threshold = np.zeros(config.kfold)
    kf = model_selection.KFold(n_splits= config.kfold, random_state= config.kfold_seed,  shuffle=True)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_data['z'])):
        fold_start = time.time()
        FoldTrain, FoldValid = train_data.iloc[train_index, :], train_data.iloc[valid_index, :]

        # upsample
        with utils.timer('Upsample on train'):
            X_train = np.array(FoldTrain['images'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))
            Y_train = np.array(FoldTrain['masks'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))
            X_valid = np.array(FoldValid['images'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))
            Y_valid = np.array(FoldValid['masks'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))

        # data augmentation, just for train part
        with utils.timer('Augmentation on train'):
            X_train = data_utils.y_axis_flip(X_train)
            Y_train = data_utils.y_axis_flip(Y_train)

        print('\n ---- Sanity check for input shape ----')
        print(X_train.shape)
        print(Y_train.shape)
        print('\n')

        model_weight_file = '%s/%s.weight.%s' % (ModelWeightDir, config.strategy, fold)

        iou = -1
        threshold = 0.0
        if(config.strategy == 'unet'):
            # initialize model
            model = UNet.UNetModel(img_shape= input_shape, start_ch= 16, depth= 5, batch_norm= True)
            # fitting
            with utils.timer('Fitting model'):
                model.fit(X_train, Y_train,X_valid, Y_valid,config.epochs, config.batch_size, model_weight_file)
            # evaluate
            with utils.timer('Evaluate model'):
                iou, threshold = model.evaluate(X_valid, np.array(FoldValid['masks'].values.tolist()).reshape((-1, config.img_size_original, config.img_size_original)))
        fold_end = time.time()

        cv_iou[fold] = iou
        cv_threshold[fold] = threshold

        print('\n ========= Summary ======== ')
        print('fold #%s: iou %.6f, threshold %.6f, time elapsed %s[s]' % (fold, iou, threshold, int(fold_end - fold_start)))
        print('============================\n')

        del FoldTrain, FoldValid, X_train, Y_train, X_valid, Y_valid
        gc.collect()
    print('\n CV IOU %.6f' % np.mean(cv_iou))
    # save evaluation result
    with open(EvaluateFile, 'w') as o_file:
        for fold in range(config.kfold):
            o_file.write('%s,%s,%s\n' % (fold, cv_iou[fold], cv_threshold[fold]))
    o_file.close()


def submit(test_data, SubmitDir, ModelWeightDir, EvaluateFile):
    ''''''
    # load evaluate result
    cv_iou = np.zeros(config.kfold, dtype= np.float32)
    cv_threshold = np.zeros(config.kfold, dtype= np.float32)
    with open(EvaluateFile, 'r') as i_file:
        for line in i_file:
            line = line.rstrip()
            if(not line):
                continue
            parts = line.split(',')
            fold = int(parts[0])
            iou = np.float32(parts[1])
            threshold = np.float32(parts[2])
            cv_iou[fold] = iou
            cv_threshold[fold] = threshold
    i_file.close()

    pred_result = np.zeros((len(test_data), config.img_size_original, config.img_size_original), dtype= np.float64)
    # do submit with CV
    for fold in range(config.kfold):

        # load model
        with utils.timer('Load model'):
            model = UNet.UNetModel(img_shape=input_shape, start_ch=16, depth=5, batch_norm=True)
            ModelWeightFile = '%s/%s.weight.%s' % (ModelWeightDir, config.strategy, fold)
            model.load_weight(ModelWeightFile)

        # upsample
        with utils.timer('Upsample'):
            X_test = np.array(test_data['images'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))

        # infer
        with utils.timer('Infer'):
            preds_test = model.predict(X_test)
            pred_result += np.array([np.round(utils.downsample(preds_test[i] > cv_threshold[fold])).tolist() for i, idx in enumerate(tqdm(test_data.index.values))])

        print('fold %s done' % fold)

    pred_result = np.round(pred_result / config.kfold)

    # save
    with utils.timer('Save'):
        pred_dict = {idx: utils.RLenc(pred_result[i]) for i, idx in enumerate(tqdm(test_data.index.values))}
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('%s/%s_submit.csv' % (SubmitDir, config.strategy))

def resubmit(TestImageDir, SubmitDir):
    ''''''
    submit_df = pd.read_csv('%s/%s_submit.csv' % (SubmitDir, config.strategy))
    for i in tqdm(range(submit_df.shape[0])):
        if str(submit_df.loc[i, 'rle_mask']) != str(np.nan):
            decoded_mask = utils.rle_decode(submit_df.loc[i, 'rle_mask'])
            orig_img = imread('%s/%s.png' % (TestImageDir, submit_df.loc[i, 'id']))
            crf_output = utils.crf(orig_img, decoded_mask)
            submit_df.loc[i, 'rle_mask'] = utils.rle_encode(crf_output)
    submit_df.to_csv('%s/%s_crf_submit.csv' % (SubmitDir, config.strategy), index=False)

if __name__ == '__main__':
    ''''''
    mode = 'train'

    RawInputDir = '%s/raw' % config.DataBaseDir
    ModelWeightDir = '%s/%s/weight' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(ModelWeightDir) == False):
        os.makedirs(ModelWeightDir)

    EvaluateFile = '%s/%s/eval.txt' % (config.ModelRootDir, config.strategy)

    SubmitDir = '%s/%s/submit' % (config.ModelRootDir, config.strategy)

    if(mode == 'train'):
        # load raw data set
        with utils.timer('Load raw data set'):
            train_data = data_utils.load_raw_train(RawInputDir)
        # train with CV
        train(train_data, ModelWeightDir, EvaluateFile)
    elif(mode == 'submit'):
        if(os.path.exists(SubmitDir) == False):
            os.makedirs(SubmitDir)
        # load test data set
        with utils.timer('Load raw test data set'):
            test_data = data_utils.load_raw_test(RawInputDir)
        submit(test_data, SubmitDir, ModelWeightDir, EvaluateFile)
    elif(mode == 'resubmit'):
        TestImageDir = '%s/test/images' % RawInputDir
        #SubmitFile = '%s/%s_submit.csv' % (SubmitDir, config.strategy)
        # resubmit with a version smoothed by CRF
        resubmit(TestImageDir, SubmitDir)
