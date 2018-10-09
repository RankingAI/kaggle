# Created by yuanpingzhou at 9/27/18

import config
import numpy as np
import glob
import pandas as pd
import matplotlib.pylab as plt
import utils
import datetime
from tqdm import tqdm
import os,sys

strategies = ['deeplab_v3', 'unet_xception', 'unet_resnet_v2']
datestr = datetime.datetime.now().strftime("%Y%m%d")

def add(x):
    return np.round((x['deeplab_v3_masks'] + x['unet_xception_masks'] + x['unet_resnet_v2_masks'])/3)
if __name__ == '__main__':
    ''''''
    predict = {}
    ids = []
    for i in range(len(strategies)):
        s = strategies[i]
        test_dir = '%s/%s/submit/test' % (config.ModelRootDir, s)
        files = glob.glob('%s/*.npy' % test_dir)
        id_list = [f.split('/')[-1].split('.')[0] for f in files]
        id_list = sorted(id_list)
        if(i == 0):
            ids = id_list
        predict['%s_masks' % s] = [np.load('%s/%s.npy' % (test_dir, id)) for id in id_list]
    predict['merged'] = [(((predict['deeplab_v3_masks'][i] + predict['unet_xception_masks'][i] + predict['unet_resnet_v2_masks'][i])/3) > 0.5).astype(np.int32)  for i in range(len(ids))]

    # visualization for demos
    plt.imsave('demo_1.jpg', predict['deeplab_v3_masks'][10])
    plt.imsave('demo_2.jpg', predict['unet_xception_masks'][10])
    plt.imsave('demo_3.jpg', predict['merged'][10])

    # save
    merge_output_dir = '%s/merge' % config.ModelRootDir
    if(os.path.exists(merge_output_dir) == False):
        os.makedirs(merge_output_dir)
    with utils.timer('save submit'):
        pred_dict = {idx: utils.RLenc(predict['merged'][i]) for i, idx in enumerate(tqdm(ids))}
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('%s/submit_%s.csv' % (merge_output_dir, datestr))
