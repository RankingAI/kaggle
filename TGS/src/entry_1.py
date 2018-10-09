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
import psutil
from data_augmentation import reflect_pad_image

os.environ['PYTHONHASHSEED'] = '0'

from skimage.io import imread, imsave

import random
import tensorflow as tf
import keras.backend as K

np.random.seed(27)
tf.set_random_seed(27)

import config
import data_utils
import plot_utils
import utils
import metric_1

import UNetWithResBlock
import UNetInceptionResnetV2
import UNetVGG16
import UNetResNet50VGG16
import UNetXception
import DeeplabV3

process = psutil.Process(os.getpid())
def _print_memory_usage():
    ''''''
    print('\n---- Current memory usage %sM ----\n' % int(process.memory_info().rss/(1024*1024)))

# configuration for GPU resources
#with K.tf.device('/device:GPU:0'):
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.9, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

#datestr = datetime.datetime.now().strftime("%Y%m%d")
datestr= '20181008'

def get_model(strategy, phase= 'train'):
    if (strategy == 'deeplab_v3'):
        model = DeeplabV3.DeeplabV3(
            input_shape=[config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3],
            stages=config.stages[strategy],
            freeze_till_layer=config.freeze_till_layer[strategy],
            print_network=False,
            phase= phase,
        )
    elif (strategy == 'unet_resnet_v2'):
        model = UNetInceptionResnetV2.UNetWithResNet(
            input_shape=[config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3],
            stages=config.stages[strategy],
            freeze_till_layer=config.freeze_till_layer[strategy],
            print_network=False,
            phase= phase,
        )
    elif (strategy == 'unet_xception'):
        model = UNetXception.UNetXception(
            input_shape=[config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3],
            stages=config.stages[strategy],
            freeze_till_layer=config.freeze_till_layer[strategy],
            print_network=False,
            phase= phase,
        )
    elif(strategy == 'unet_vgg16'):
        model = UNetVGG16.UNetVGG16(
            input_shape= [config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3],
            stages= config.stages[strategy],
            learning_rate= config.learning_rate[strategy],
            freeze_till_layer= config.freeze_till_layer[strategy],
            print_network= False,
        )
    elif(strategy == 'unet_resnet50_vgg16'):
        model = UNetResNet50VGG16.UNetResNet50VGG16(
            input_shape= [config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3],
            stages= config.stages[strategy],
            learning_rate= config.learning_rate[strategy],
            freeze_till_layer= config.freeze_till_layer[strategy],
            print_network= False,
        )
    elif (strategy == 'unet_res_block'):
        if((config.grayscale[strategy] == True) & (config.depth_channel == True)):
            input_shape = [config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3]
        elif(config.grayscale[strategy] == True):
            input_shape = [config.encoder_input_size[strategy], config.encoder_input_size[strategy], 1]
        else:
            input_shape = [config.encoder_input_size[strategy], config.encoder_input_size[strategy], 3]
        model = UNetWithResBlock.UNetWithResBlock(input_shape= input_shape,
                                                  stages=config.stages[strategy],
                                                  print_network=False,
                                                  )
    else:
        print('=========== strategy %s not matched!!!' % strategy)
        return None
    return model

def get_aug_params(strategy):
    ''''''
    if(config.grayscale[strategy] == False):
        channels = 3
    elif((config.grayscale[strategy] == True) & (config.depth_channel == True)):
        channels = 3
    else:
        channels = 1
    double_size = False
    pad_size_1 = 0
    pad_size_2 = 0
    if(config.encoder_input_size[strategy] == 128):
        pad_size_1 = 13
        pad_size_2 = 14
        double_size = False
    elif(config.encoder_input_size[strategy] == 256):
        pad_size_1 = 27
        pad_size_2 = 27
        double_size = True

    return channels, double_size, pad_size_1, pad_size_2

def train(train_data, ModelWeightDir, EvaluateFile, image_files, PredictDir, strategy):
    ''''''
    # save evaluation result
    eval_f = open(EvaluateFile, 'w')

    # CV
    cv_train = np.zeros((len(train_data), config.img_size_original, config.img_size_original), dtype= np.int32)
    cv_fold = np.zeros(len(train_data))

    cv_iou = np.zeros((config.kfold, config.stages[strategy]))
    cv_threshold = np.zeros((config.kfold, config.stages[strategy]))

    _print_memory_usage()

    channels, double_size, pad_size_1, pad_size_2 = get_aug_params(strategy)

    ## cv with depth version
    kf = model_selection.KFold(n_splits= config.kfold, random_state= config.kfold_seed, shuffle= True)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_data['z'])):
        print('\n ---- Fold %s starts... \n' % fold)

        fold_start = time.time()

        FoldTrain, FoldValid = train_data.iloc[train_index, :], train_data.iloc[valid_index, :]

        # resize
        with utils.timer('resize'):
            if(double_size):
                X_train = [utils.img_resize(v, config.img_size_original, config.img_size_original * 2) for v in FoldTrain['images'].values]
                Y_train = [utils.img_resize(v, config.img_size_original, config.img_size_original * 2) for v in FoldTrain['masks'].values]
                X_valid = [utils.img_resize(v, config.img_size_original, config.img_size_original * 2) for v in FoldValid['images'].values]
                Y_valid = [utils.img_resize(v, config.img_size_original, config.img_size_original * 2) for v in FoldValid['masks'].values]
            else:
                X_train = FoldTrain['images'].values
                Y_train = FoldTrain['masks'].values
                X_valid = FoldValid['images'].values
                Y_valid = FoldValid['masks'].values

            print('image size: ')
            print(X_train[0].shape)

        # reflect padding
        with utils.timer('padding'):
            if((pad_size_1 > 0) | (pad_size_2 > 0)):
                X_train = np.array([reflect_pad_image(v, pad_size_1, pad_size_2).tolist() for v in X_train])
                Y_train = np.array([reflect_pad_image(v, pad_size_1, pad_size_2).tolist() for v in Y_train])
                X_valid = np.array([reflect_pad_image(v, pad_size_1, pad_size_2).tolist() for v in X_valid])
                Y_valid = np.array([reflect_pad_image(v, pad_size_1, pad_size_2).tolist() for v in Y_valid])
            else:
                X_train = np.array([v.tolist() for v in X_train])
                Y_train = np.array([v.tolist() for v in Y_train])
                X_valid = np.array([v.tolist() for v in X_valid])
                Y_valid = np.array([v.tolist() for v in Y_valid])

            print('image size: ')
            print(X_train[0].shape)

        # add depth channels if possible
        with utils.timer('add depth channels'):
            if((config.grayscale[strategy] == True) & (config.depth_channel == True)):
                X_train = np.array([data_utils.add_depth_channels(np.array(v)).tolist() for v in X_train])
                X_valid = np.array([data_utils.add_depth_channels(np.array(v)).tolist() for v in X_valid])

                print('image size: ')
                print(X_train[0].shape)

        # augmentation, horizontal flip
        with utils.timer('flip'):
            X_train = data_utils.y_axis_flip(X_train).reshape((-1, config.encoder_input_size[strategy], config.encoder_input_size[strategy], channels))
            Y_train = data_utils.y_axis_flip(Y_train).reshape((-1, config.encoder_input_size[strategy], config.encoder_input_size[strategy], 1))
            X_valid = X_valid.reshape((-1, config.encoder_input_size[strategy], config.encoder_input_size[strategy], channels))
            Y_valid = Y_valid.reshape((-1, config.encoder_input_size[strategy], config.encoder_input_size[strategy], 1))

            print('image size: ')
            print(X_train[0].shape)

        _print_memory_usage()

        print('\n ---- Sanity check for input shape ----')
        print('shape of X_train: ')
        print(X_train.shape)
        print('shape of Y_train:')
        print(Y_train.shape)
        print('shape of X_valid:')
        print(X_valid.shape)
        print('shape of Y_valid:')
        print(Y_valid.shape)
        print('\n')

        model_weight_file = '%s/%s.fold.%s' % (ModelWeightDir, strategy, fold)
        model = get_model(strategy)

        for s in range(config.stages[strategy]):
            print('training on stage %s ...' % s)

            stage_model_weight_file = '{}.stage.{}'.format(model_weight_file, s)

            # fitting
            with utils.timer('Fitting model %s' % s):
                model.fit(X_train, Y_train, X_valid, Y_valid,
                          epochs= config.epochs[strategy][s],
                          batch_size= config.batch_size[strategy],
                          model_weight_file= stage_model_weight_file,
                          learning_rate= config.learning_rate[strategy][s],
                          stage= s,
                          snapshot_ensemble= False)

            # evaluate
            with utils.timer('Evaluate with model %s' % s):
                # predict
                pred_valid = model.predict(X_valid, stage= s)
                # restore into original shape
                if((pad_size_1 > 0) | (pad_size_2 > 0)):
                    pred_valid = [v[pad_size_1: -pad_size_2, pad_size_1: -pad_size_2] for v in pred_valid]
                if(double_size):
                    pred_valid = np.array([utils.img_resize(v, from_size= config.img_size_original * 2, to_size= config.img_size_original).squeeze().tolist() for v in pred_valid])
                else:
                    pred_valid = np.array([v.squeeze().tolist() for v in pred_valid])
                print('predict done.')
                # evaluate
                iou, threshold = model.evaluate(pred_valid, np.array(FoldValid['masks'].values.tolist()).reshape((-1, config.img_size_original, config.img_size_original)), stage= s)
                cv_iou[fold, s] = iou
                cv_threshold[fold, s] = threshold
                print('evaluate done.')

            if(s == config.stages[strategy] - 1):
                cv_train[valid_index] = pred_valid
                cv_fold[valid_index] = fold

            _print_memory_usage()

        fold_end = time.time()

        print('\n ========= Summary ======== ')
        for s in range(config.stages[strategy]):
            print('fold #%s, stage %s/%s: iou %.6f, threshold %.6f, time elapsed %s[s]' % (fold, s, config.stages[strategy], cv_iou[fold, s], cv_threshold[fold, s], int(fold_end - fold_start)))
        print('============================\n')

        iou_str = ','.join(['%.6f' % v for v in cv_iou[fold, :]])
        thre_str = ','.join(['%.6f' % v for v in cv_threshold[fold, :]])

        eval_f.write('%s,%s,%s\n' % (fold, iou_str, thre_str))
        eval_f.flush()

        del FoldTrain, FoldValid, X_train, Y_train, X_valid, Y_valid, model
        gc.collect()

        _print_memory_usage()

        with utils.timer('clean session'):
            K.clear_session()

        _print_memory_usage()

    print('\n CV IOU %.6f' % np.mean(cv_iou[:,-1]))
    eval_f.close()

    _print_memory_usage()

    # save prediction on train data set for debug
    PredictMaskDir = '%s/masks' % PredictDir
    if(os.path.exists(PredictMaskDir) == False):
        os.makedirs(PredictMaskDir)
    with open('%s/cv_fold_%s.txt' % (PredictDir, datestr), 'w') as o_file:
        for i in range(len(image_files)):
            image_name = image_files[i].split('/')[-1].split('.')[0]
            np.save('%s/%s.npy' % (PredictMaskDir, image_name), cv_train[i])
            o_file.write('%s,%s\n' % (image_name, cv_fold[i]))
    o_file.close()

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def infer(test_data, ModelWeightDir, EvaluateFile, strategy):
    ''''''
    # load evaluate result
    cv_iou = np.zeros((config.kfold, config.stages[strategy]), dtype= np.float32)
    cv_threshold = np.zeros((config.kfold, config.stages[strategy]), dtype= np.float32)
    with open(EvaluateFile, 'r') as i_file:
        for line in i_file:
            line = line.rstrip()
            if(not line):
                continue
            parts = line.split(',')
            fold = int(parts[0])
            for s in range(config.stages[strategy]):
                cv_iou[fold, s] = np.float32(parts[1 + s])
            for s in range(config.stages[strategy]):
                cv_threshold[fold, s] = np.float32(parts[1 + config.stages[strategy] * 1 + s])
    i_file.close()
    best_stage = np.argmax(cv_iou, axis= 1)
    best_iou = cv_iou[np.arange(len(cv_iou)), best_stage]
    best_threshold = cv_threshold[np.arange(len(cv_threshold)), best_stage]
    print('\n------------------------------------')
    print('best stage in folds:')
    print(best_stage)
    print('best iou in folds:')
    print(best_iou)
    print('best threshold in folds:')
    print(best_threshold)
    print('------------------------------------\n')

    with utils.timer('reset index on test data'):
        test_data.reset_index(drop= False, inplace= True)
        test_data.rename(index= str, columns= {'index': 'id'}, inplace= True)

    channels, double_size, pad_size_1, pad_size_2 = get_aug_params(strategy)

    pred_result = np.zeros((len(test_data), config.img_size_original, config.img_size_original), dtype= np.float64)
    # do submit with CV
    for fold in range(config.kfold):

        # load model
        with utils.timer('Load model'):
            # model architecture
            model= get_model(strategy, 'submit')
            # load weight
            ModelWeightFile = '%s/%s.fold.%s.stage.%s' % (ModelWeightDir, strategy, fold, best_stage[fold])
            model.load_weight(ModelWeightFile, best_stage[fold])

        print('fold %s, stage %s, threshold %.6f, iou %.6f' % (fold, best_stage[fold], best_threshold[fold], best_iou[fold]))

        # infer
        with utils.timer('Infer'):
            preds_test = []

            for batch_no, batch_index in enumerate(batch(test_data.index.values, config.infer_batch_size)):

                with utils.timer('batch %s/%s' % (((batch_no + 1) * config.infer_batch_size), len(test_data))):
                    test_images = test_data['images'][batch_index]

                    # resize
                    if (double_size):
                        with utils.timer('resize'):
                            X_test = [utils.img_resize(v, config.img_size_original, config.img_size_original * 2) for v in test_images]
                    else:
                        X_test = test_images

                    # reflect padding
                    if ((pad_size_1 > 0) | (pad_size_2 > 0)):
                        with utils.timer('padding'):
                            X_test = np.array([reflect_pad_image(v, pad_size_1, pad_size_2).tolist() for v in X_test])

                    # add depth channels if possible
                    if ((config.grayscale[strategy] == True) & (config.depth_channel == True)):
                        with utils.timer('add depth channels'):
                            X_test = np.array([data_utils.add_depth_channels(np.array(v)).tolist() for v in X_test])

                    X_test = X_test.reshape((-1, config.encoder_input_size[strategy], config.encoder_input_size[strategy], channels))

                    # predict value, logit or probability, with the last model
                    pred_batch = model.predict(X_test, stage= best_stage[fold])

                    # restore into original shape
                    if ((pad_size_1 > 0) | (pad_size_2 > 0)):
                        with utils.timer('padding'):
                            pred_batch = [v[pad_size_1: -pad_size_2, pad_size_1: -pad_size_2] for v in pred_batch]
                    if (double_size):
                        with utils.timer('resize'):
                            pred_batch = [utils.img_resize(v, from_size=config.img_size_original * 2,to_size=config.img_size_original) for v in pred_batch]
                    print('predict done.')

                    # resize to original shape and convert into label
                    pred_batch = [np.int32(np.round(v > best_threshold[fold])).squeeze().tolist() for v in pred_batch]
                    preds_test.append(pred_batch)
                    del X_test, pred_batch
                    gc.collect()
            pred_result += np.array(preds_test).reshape((len(test_data), config.img_size_original, config.img_size_original))
            del preds_test
            gc.collect()

        print('fold %s done' % fold)
        with utils.timer('clear session'):
            K.clear_session()

    # average them
    pred_result = np.round(pred_result / config.kfold)
    #pred_result = pred_result

    return pred_result

def save_submit(predict, indexes, SubmitDir, strategy):
    #
    TestOutput = '%s/test' % SubmitDir
    if(os.path.exists(TestOutput) == False):
        os.makedirs(TestOutput)
    with utils.timer('save test'):
        for i, idx in enumerate(tqdm(indexes)):
            np.save('%s/%s.npy' % (TestOutput, idx), predict[i])

    # save
    with utils.timer('save submit'):
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

    parser.add_argument('-strategy', "--strategy",
                        default= 'unet_resnet_v2',
                        help= "strategy",
                        choices= ['unet_res_block', 'unet_resnet_v2', 'unet_vgg16', 'unet_resnet50_vgg16', 'unet_xception', 'deeplab_v3'])

    parser.add_argument('-phase', "--phase",
                        default= 'submit',
                        help= "project phase",
                        choices= ['train', 'debug', 'submit', 'resubmit'])
    parser.add_argument('-data_input', '--data_input',
                        default= '%s/raw' % (config.DataBaseDir)
                        )
    args = parser.parse_args()

    model_output = '%s/%s' % (config.ModelRootDir, args.strategy)

    # evaluation log
    EvaluateFile = '%s/eval_%s.txt' % (model_output, datestr)

    # model weight
    ModelWeightDir = '%s/weight' % model_output
    if(os.path.exists(ModelWeightDir) == False):
        os.makedirs(ModelWeightDir)

    # for debugging
    PredictDir = '%s/predict' % model_output
    if(os.path.exists(PredictDir) == False):
        os.makedirs(PredictDir)

    # for submit
    SubmitDir = '%s/submit' % model_output
    if(os.path.exists(SubmitDir) == False):
        os.makedirs(SubmitDir)

    if(args.phase == 'train'):
        # load raw data set
        with utils.timer('Load raw data set'):
            train_data, image_files = data_utils.load_raw_train(args.data_input, return_image_files= True, debug= config.debug, grayscale= config.grayscale[args.strategy])
        # train with CV
        train(train_data, ModelWeightDir, EvaluateFile, image_files, PredictDir, args.strategy)
    elif(args.phase == 'debug'):
        sanity_check(args.data_input, '%s/train' % args.data_input, PredictDir, model_output)
    elif(args.phase == 'submit'):
        # load test data set
        with utils.timer('Load raw test data set'):
            test_data = data_utils.load_raw_test(args.data_input, grayscale= config.grayscale[args.strategy])
        pred_test = infer(test_data, ModelWeightDir, EvaluateFile, args.strategy)
        save_submit(pred_test, test_data['id'].values, SubmitDir, args.strategy)
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
        ModelOutpuDir = '%s/%s' % (config.ModelRootDir, args.strategy)
        SubmitDir = '%s/submit' % ModelOutpuDir
        if(os.path.exists(SubmitDir) == False):
            os.makedirs(SubmitDir)
        resubmit(TestImageDir, SubmitDir, args.strategy)
