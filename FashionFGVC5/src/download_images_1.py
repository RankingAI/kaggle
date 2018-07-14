import json
import wget
import os,sys,time,datetime
import pandas as pd
import psutil
from contextlib import contextmanager
import threading

DataBaseDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/FashionFGVC5/data'

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
    ImageRootOutputDir = '%s/raw/images/%s' % (DataBaseDir, mod)
    if(os.path.exists(ImageRootOutputDir) == False):
        os.makedirs(ImageRootOutputDir)
    FailedRootOutputDir = '%s/raw/failed/%s' % (DataBaseDir, mod)
    if (os.path.exists(FailedRootOutputDir) == False):
        os.makedirs(FailedRootOutputDir)

    json_data['images'] = json_data['images'][:51]

    class DownloadThread(threading.Thread):
        def __init__(self, t, low, high):
            super(DownloadThread, self).__init__()
            self.no = t
            self.low = low
            self.high = high
            self.FailedFile = '%s/%s.txt' % (FailedRootOutputDir, self.no)
            ## save images
            self.ImageOutputDir = '%s/%s' % (ImageRootOutputDir, self.no)
            if (os.path.exists(self.ImageOutputDir) == False):
                os.makedirs(self.ImageOutputDir)

        def run(self):
            with open(self.FailedFile, 'w') as w_file:
                for i in range(self.low, self.high):
                    try:
                        #print('%s/%s.jpg' % (ImageOutputDir, json_data['images'][i]['imageId']))
                        wget.download(json_data['images'][i]['url'], out= '%s/%s.jpg' % (self.ImageOutputDir, json_data['images'][i]['imageId']))
                        print('%s, %s, %s, %s' % (self.no, i, self.low, self.high))
                    except:
                        w_file.write('%s,%s\n' % (json_data['images'][i]['url'], json_data['images'][i]['imageId']))

    num_thread = 4
    block_size = int(len(json_data['images'])/num_thread)
    print(block_size)
    threads = []
    for t in range(num_thread):
        if((t + 1) * block_size > len(json_data['images'])):
            d_thread = DownloadThread(t, t * block_size, len(json_data['images']))
        else:
            d_thread = DownloadThread(t, t * block_size, (t + 1) * block_size)
        threads.append(d_thread)
    for thre in threads:
        thre.start()
    for thre in threads:
        thre.join()

    break
