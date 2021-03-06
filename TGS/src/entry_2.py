#################################
# End-2-End pipeline for models #
#################################
import os, sys, time, gc, datetime
import numpy as np
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
import argparse
import glob

os.environ['PYTHONHASHSEED'] = '0'

from skimage.io import imread, imsave

import random
import tensorflow as tf
import keras.backend as K

np.random.seed(27)
#random.random.seed(27)
tf.set_random_seed(27)

import config
import data_utils
import plot_utils
import utils
import UNetWithResBlock
import metric_1

# configuration for GPU resources
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

input_shape = (config.img_size_target,config.img_size_target,1)
datestr = datetime.datetime.now().strftime("%Y%m%d")

def train(train_data, ModelWeightDir, EvaluateFile, image_files, PredictDir):
    ''''''
    # CV
    cv_train = np.zeros((len(train_data), config.img_size_original, config.img_size_original), dtype= np.int32)
    cv_iou = np.zeros(config.kfold)
    cv_threshold = np.zeros(config.kfold)
    cv_fold = np.zeros(len(train_data))
    ## cv with depth version
    kf = model_selection.KFold(n_splits= config.kfold, random_state= config.kfold_seed, shuffle= True)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_data['z'])):
        fold_start = time.time()
        FoldTrain, FoldValid = train_data.iloc[train_index, :], train_data.iloc[valid_index, :]

        ## upsample
        #with utils.timer('Upsample on train'):
        #    X_train = np.array(FoldTrain['images'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))
        #    Y_train = np.array(FoldTrain['masks'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))
        #    X_valid = np.array(FoldValid['images'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))
        #    Y_valid = np.array(FoldValid['masks'].map(utils.upsample).tolist()).reshape((-1, config.img_size_target, config.img_size_target, 1))

        # data augmentation, just for train part
        with utils.timer('Augmentation on train'):
            X_train = data_utils.y_axis_flip(np.array(FoldTrain['images'].tolist()).reshape((-1, config.img_size_original, config.img_size_original, 1)))
            Y_train = data_utils.y_axis_flip(np.array(FoldTrain['masks'].tolist()).reshape((-1, config.img_size_original, config.img_size_original, 1)))
            X_valid = np.array(FoldValid['images'].tolist()).reshape((-1, config.img_size_original, config.img_size_original, 1))
            Y_valid = np.array(FoldValid['masks'].tolist()).reshape((-1, config.img_size_original, config.img_size_original, 1))

        print('\n ---- Sanity check for input shape ----')
        print(X_train.shape)
        print(Y_train.shape)
        print('\n')

        model_weight_file = '%s/%s.weight.%s' % (ModelWeightDir, config.strategy, fold)

        iou = -1
        threshold = 0.0
        if(config.strategy == 'unet_res_block_depth'):
            # initialize model
            model = UNetWithResBlock.UNetWithResBlock(print_network= False)
            # fitting
            with utils.timer('Fitting model'):
                model.fit(X_train, Y_train, X_valid, Y_valid, config.epochs, config.batch_size, model_weight_file)
            # evaluate
            with utils.timer('Evaluate model'):
                pred_valid = model.predict(X_valid)
                iou, threshold = model.evaluate(pred_valid, np.array(FoldValid['masks'].values.tolist()).reshape((-1, config.img_size_original, config.img_size_original)))
                cv_train[valid_index] = pred_valid
                cv_fold[valid_index] = fold
        else:
            print('=========== strategy %s not matched!!!' % config.strategy)
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
    # save prediction on train data set
    PredictMaskDir = '%s/masks' % PredictDir
    if(os.path.exists(PredictMaskDir) == False):
        os.makedirs(PredictMaskDir)
    with open('%s/cv_fold_%s.txt' % (PredictDir, datestr), 'w') as o_file:
        for i in range(len(image_files)):
            image_name = image_files[i].split('/')[-1].split('.')[0]
            np.save('%s/%s.npy' % (PredictMaskDir, image_name), cv_train[i])
            o_file.write('%s,%s\n' % (image_name, cv_fold[i]))
    o_file.close()

def infer(test_data, ModelWeightDir, EvaluateFile, strategy):
    ''''''
    # load evaluate result
    #cv_iou = np.zeros(config.kfold, dtype= np.float32)
    #cv_threshold = np.zeros(config.kfold, dtype= np.float32)
    #with open(EvaluateFile, 'r') as i_file:
    #    for line in i_file:
    #        line = line.rstrip()
    #        if(not line):
    #            continue
    #        parts = line.split(',')
    #        fold = int(parts[0])
    #        iou = np.float32(parts[1])
    #        threshold = np.float32(parts[2])
    #        cv_iou[fold] = iou
    #        cv_threshold[fold] = threshold
    #i_file.close()

    pred_result = np.zeros((len(test_data), config.img_size_original, config.img_size_original), dtype= np.float64)
    # do submit with CV
    for fold in range(config.kfold):

        # load model
        with utils.timer('Load model'):
            model = UNetWithResBlock.UNetWithResBlock(print_network= False)
            ModelWeightFile = '%s/%s.weight.%s' % (ModelWeightDir, strategy, fold)
            model.load_weight(ModelWeightFile)

        # infer
        with utils.timer('Infer'):
            preds_test = model.predict(np.array(test_data['images'].tolist()).reshape((-1, config.img_size_original, config.img_size_original, 1)))
            #pred_result += np.array([np.int32(preds_test[i] > cv_threshold[fold]).tolist() for i in tqdm(range(len(pred_test)))])
            pred_result += np.array([np.int32(preds_test[i] > 0.406667).tolist() for i in tqdm(range(len(preds_test)))])

        print('fold %s done' % fold)
        break

    #pred_result = np.round(pred_result / config.kfold)

    return pred_result

def save_submit(predict, indexes, SubmitDir, strategy):
    # save
    with utils.timer('Save'):
        pred_dict = {idx: utils.RLenc(predict[i]) for i, idx in enumerate(tqdm(indexes))}
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('%s/%s_submit_%s.csv' % (SubmitDir, strategy, datestr))

def resubmit(TestImageDir, SubmitDir, strategy):
    ''''''
    submit_df = pd.read_csv('%s/%s_submit_%s.csv' % (SubmitDir, strategy, datestr))
    for i in tqdm(range(submit_df.shape[0])):
        if str(submit_df.loc[i, 'rle_mask']) != str(np.nan):
            decoded_mask = utils.rle_decode(submit_df.loc[i, 'rle_mask'])
            orig_img = imread('%s/%s.png' % (TestImageDir, submit_df.loc[i, 'id']))
            crf_output = utils.crf(orig_img, decoded_mask)
            submit_df.loc[i, 'rle_mask'] = utils.rle_encode(crf_output)
    submit_df.to_csv('%s/%s_crf_submit_%s.csv' % (SubmitDir, strategy, datestr), index=False)

def image2df(files):
    images = [(f.split('/')[-1].split('.')[0], imread(f, flatten= True)) for f in files]
    df = pd.DataFrame(index= range(len(images)))
    df['img_name'] = [v[0] for v in images]
    df['img_data'] = [v[1] for v in images]

    return df

def array2df(files):
    images = [(f.split('/')[-1].split('.')[0], np.load(f)) for f in files]
    df = pd.DataFrame(index= range(len(images)))
    df['img_name'] = [v[0] for v in images]
    df['img_data'] = [v[1] for v in images]

    return df

def sanity_check(depth_dir, truth_dir, predict_dir, output_dir):
    # depth
    depth_df = pd.read_csv('%s/depths.csv' % depth_dir)
    depth_df['img_name'] = depth_df['id']
    depth_df.drop(['id'], axis= 1, inplace= True)
    # images
    image_files = glob.glob('%s/images/*.png' % truth_dir)
    image_df = image2df(image_files)
    #
    image_fold_df = pd.read_csv('%s/cv_fold_20180906.txt' % predict_dir, header= None)
    image_fold_df.columns = ['img_name', 'fold']
    image_fold_df['fold'] = image_fold_df['fold'].astype(np.int32)
    #
    eval_df = pd.read_csv('%s/eval_20180906.txt' % '/'.join(predict_dir.split('/')[:-1]), header= None)
    eval_df.columns = ['fold', 'iou', 'threshold']
    eval_df['fold'] = eval_df['fold'].astype(np.int32)
    image_fold_df = pd.merge(image_fold_df, eval_df, how= 'left', on= 'fold')
    image_df = pd.merge(image_df, image_fold_df, how= 'left', on= 'img_name')
    print('\n -------- image fold --------')
    print(image_fold_df.head(10))
    print('----------------\n')
    # truth mask
    truth_mask_files = glob.glob('%s/masks/*.png' % truth_dir)
    truth_mask_df = image2df(truth_mask_files)
    truth_mask_df['mask_data'] = truth_mask_df['img_data']
    truth_mask_df.drop(['img_data'], axis= 1, inplace= True)
    image_df = pd.merge(image_df, truth_mask_df, how= 'left', on= 'img_name')
    # check
    print('\n -------- checking for merging ------- ')
    print(image_df.isnull().sum(axis= 0))
    print('----------------\n')
    # predict mask
    pred_mask_files = glob.glob('%s/masks/*.npy' % predict_dir)
    pred_mask_df = array2df(pred_mask_files)
    pred_mask_df['pred_mask_data'] = pred_mask_df['img_data']
    pred_mask_df.drop(['img_data'], axis= 1, inplace= True)
    image_df = pd.merge(image_df, pred_mask_df, how= 'left', on= 'img_name')
    # check
    print('\n -------- checking for merging ------- ')
    print(image_df.isnull().sum(axis= 0))
    print('-----------------\n')
    #
    image_df = pd.merge(image_df, depth_df, how= 'left', on= 'img_name')
    print('merge done!!!')
    #
    image_df['prec'] = image_df.apply(lambda x: metric_1.iou_metric(x['mask_data'], (x['pred_mask_data'] > x['threshold']).astype(np.int32)), axis= 1)
    image_df['mask_coverage_ratio'] = image_df['mask_data'].apply(lambda  x: np.sum(x)) / pow(config.img_size_original, 2)
    image_df['pred_mask_coverage_ratio'] = image_df.apply(lambda x: np.sum((x['pred_mask_data'] > x['threshold']).astype(np.int32)), axis= 1) / pow(config.img_size_original, 2)
    print(image_df[['prec', 'iou']].head(50))
    #
    plot_utils.truth_vs_predict_mask(image_df, '%s/demo_1.jpg' % output_dir)

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()

    parser.add_argument('-phase', "--phase",
                        default= 'debug',
                        help= "project phase",
                        choices= ['train', 'debug', 'submit', 'resubmit'])
    parser.add_argument('-data_input', '--data_input',
                        default= '%s/raw' % (config.DataBaseDir)
                        )
    parser.add_argument('-model_output', '--model_output',
                        default= '%s/%s' % (config.ModelRootDir, config.strategy),
                        )
    args = parser.parse_args()

    # evaluation log
    EvaluateFile = '%s/eval_%s.txt' % (args.model_output, datestr)

    # model weight
    ModelWeightDir = '%s/weight' % args.model_output
    if(os.path.exists(ModelWeightDir) == False):
        os.makedirs(ModelWeightDir)

    # for debugging
    PredictDir = '%s/predict' % args.model_output
    if(os.path.exists(PredictDir) == False):
        os.makedirs(PredictDir)

    # for submit
    SubmitDir = '%s/submit' % args.model_output
    if(os.path.exists(SubmitDir) == False):
        os.makedirs(SubmitDir)

    if(args.phase == 'train'):
        # load raw data set
        with utils.timer('Load raw data set'):
            train_data, image_files = data_utils.load_raw_train(args.data_input, return_image_files= True)
        # train with CV
        train(train_data, ModelWeightDir, EvaluateFile, image_files, PredictDir)
    elif(args.phase == 'debug'):
        sanity_check(args.data_input, '%s/train' % args.data_input, PredictDir, args.model_output)
    elif(args.phase == 'submit'):
        # load test data set
        with utils.timer('Load raw test data set'):
            test_data = data_utils.load_raw_test(args.data_input)
        pred_test = infer(test_data, ModelWeightDir, EvaluateFile, config.strategy)
        save_submit(pred_test, test_data.index.values, SubmitDir, config.strategy)
    elif(args.phase == 'merge_submit'):
        with utils.timer('Load raw test data set'):
            test_data = data_utils.load_raw_test(args.data_input)

        ModelOutpuDir = '%s/unet_depth' % (config.ModelRootDir)
        pred_test_depth = infer(test_data, '%s/weight' % ModelOutpuDir, '%s/eval_20180906.txt' % ModelOutpuDir, 'unet_depth')

        ModelOutpuDir = '%s/unet_coverage_level' % (config.ModelRootDir)
        pred_test_coverage_level = infer(test_data, '%s/weight' % ModelOutpuDir, '%s/eval_20180906.txt' % ModelOutpuDir, 'unet_coverage_level')

        pred_test_merge = np.round((pred_test_depth + pred_test_coverage_level)/2)

        ModelOutpuDir = '%s/unet_depth_coverage_level' % config.ModelRootDir
        SubmitDir = '%s/submit' % ModelOutpuDir
        if(os.path.exists(SubmitDir) == False):
            os.makedirs(SubmitDir)
        save_submit(pred_test_merge, test_data.index.values, SubmitDir, 'unet_depth_coverage_level')
    elif(args.phase == 'resubmit'):
        TestImageDir = '%s/test/images' % args.data_input
        ModelOutpuDir = '%s/%s' % (config.ModelRootDir, config.strategy)
        SubmitDir = '%s/submit' % ModelOutpuDir
        if(os.path.exists(SubmitDir) == False):
            os.makedirs(SubmitDir)
        resubmit(TestImageDir, SubmitDir, config.strategy)
