import json
import wget
import os,sys,time,datetime
import pandas as pd
import psutil
from contextlib import contextmanager

DataBaseDir = '../data'

## timer function
@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'\n[{name}] done in {time.time() - t0:.0f} s')

for mod in ['train', 'validation', 'test']:
    ## load data
    json_file = '%s/raw/%s.json' % (DataBaseDir, mod)
    with open(json_file, 'r') as i_file:
        json_data = json.load(i_file)
    i_file.close()

    ## save labels
    if(mod != 'test'):
        LabelOutputDir = '%s/raw/labels' % (DataBaseDir)
        if(os.path.exists(LabelOutputDir) == False):
            os.makedirs(LabelOutputDir)
        labels = dict([(pair['imageId'], pair['labelId']) for pair in json_data['annotations']])
        label_df = pd.DataFrame()
        label_df['image_id'] = list(labels.keys())
        label_df['label_id'] = [','.join(l) for l in labels.values()]
        label_df[['image_id', 'label_id']].to_csv('%s/%s.csv' % (LabelOutputDir, mod), index= False)

    ## save images
    ImageOutputDir = '%s/raw/images/%s' % (DataBaseDir, mod)
    if(os.path.exists(ImageOutputDir) == False):
        os.makedirs(ImageOutputDir)
    i = 0
    times = 6
    sub_loss = 0
    total_loss = 0
    block_size = 1000
    e = 0
    FailedOutputDir = '%s/raw/failed' % DataBaseDir
    if(os.path.exists(FailedOutputDir) == False):
        os.makedirs(FailedOutputDir)
    FailedFile = '%s/%s.txt' % (FailedOutputDir, mod)
    with open(FailedFile, 'w') as w_file:
        for im_info in json_data['images']:
            try:
                wget.download(im_info['url'], out= '%s/%s.jpg' % (ImageOutputDir, im_info['imageId']))
            except:
                w_file.write('%s,%s\n' % (im_info['url'], im_info['imageId']))
                sub_loss += 1
            if(i % block_size == 0):
                print('%s done, loss %s' % (i, sub_loss))
                total_loss += sub_loss
                sub_loss = 0
            i += 1
            # if(i == 100):
            #     break
        print('\n=========== %s done, loss %s.' % (mod, total_loss))
