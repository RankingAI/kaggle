import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
import config
from tqdm import tqdm
import sys

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

def y_axis_flip(data):
    ''''''
    return np.append(data, [np.fliplr(x) for x in data], axis= 0)
