import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
import config
from tqdm import tqdm
import sys
import glob

def load_raw_train(InputDir):
    ''''''
    # id & depth
    train_df = pd.read_csv("%s/train/train.csv" % InputDir, index_col="id", usecols=[0])
    depths_df = pd.read_csv("%s/depths.csv" % InputDir, index_col="id")
    train_df = train_df.join(depths_df)
    #test_df = depths_df[~depths_df.index.isin(train_df.index)]

    # image
    train_df["images"] = [np.array(load_img("%s/train/images/%s.png" % (InputDir, idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    # mask
    train_df["masks"] = [np.array(load_img("%s/train/masks/%s.png" % (InputDir, idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    # coverage
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(config.img_size_original, 2)

    return  train_df

def load_raw_test(InputDir):
    ''''''
    depths_df = pd.read_csv("%s/depths.csv" % InputDir, index_col="id")
    test_files = glob.glob('%s/test/images/*.png' % InputDir)
    test_ids = [f.split('/')[-1].split('.')[0] for f in test_files]
    test_df = pd.DataFrame(index= test_ids)
    test_df = test_df.join(depths_df)
    print(len(depths_df), len(test_df))

    # image
    test_df["images"] = [np.array(load_img("%s/test/images/%s.png" % (InputDir, idx), grayscale=True)) / 255 for idx in tqdm(test_df.index)]

    return test_df

def y_axis_flip(data):
    ''''''
    return np.append(data, [np.fliplr(x) for x in data], axis= 0)
