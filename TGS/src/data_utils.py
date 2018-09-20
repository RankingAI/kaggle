import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
import config
from tqdm import tqdm
import sys
import glob
import math

def cov_to_class(val):
    return np.int(np.ceil(val * 10))

def load_raw_train(InputDir, return_image_files= False, debug= False, grayscale= True):
    ''''''
    # id & depth
    train_df = pd.read_csv("%s/train/train.csv" % InputDir, index_col="id", usecols=[0])
    depths_df = pd.read_csv("%s/depths.csv" % InputDir, index_col="id")
    train_df = train_df.join(depths_df)

    # image
    train_df["images"] = [np.array(load_img("%s/train/images/%s.png" % (InputDir, idx), grayscale= grayscale)) / 255 for idx in tqdm(train_df.index)]

    # mask
    train_df["masks"] = [np.array(load_img("%s/train/masks/%s.png" % (InputDir, idx), grayscale= True)) / 255 for idx in tqdm(train_df.index)]

    # coverage
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(config.img_size_original, 2)

    # coverage level
    train_df['coverage_level'] = train_df['coverage'].apply(lambda x: int(math.ceil(x * 10)))

    if(debug == True):
        train_df = train_df.sample(frac= 0.1)

    if(return_image_files):
        image_files = ['%s/train/images/%s.png' % (InputDir, idx) for idx in train_df.index.values]
        return train_df, image_files
    else:
        return  train_df

def load_raw_test(InputDir, grayscale= True):
    ''''''
    depths_df = pd.read_csv("%s/depths.csv" % InputDir, index_col="id")
    test_files = glob.glob('%s/test/images/*.png' % InputDir)
    test_ids = [f.split('/')[-1].split('.')[0] for f in test_files]
    test_df = pd.DataFrame(index= test_ids)
    test_df = test_df.join(depths_df)
    print('depth size %s, test size %s' % (len(depths_df), len(test_df)))

    # image
    test_df["images"] = [np.array(load_img("%s/test/images/%s.png" % (InputDir, idx), grayscale= grayscale)) / 255 for idx in tqdm(test_df.index)]

    return test_df

def y_axis_flip(data):
    ''''''
    return np.append(data, [np.fliplr(x) for x in data], axis= 0)
