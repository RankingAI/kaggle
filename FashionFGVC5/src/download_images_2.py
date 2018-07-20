import pandas as pd
import numpy as np
import os,sys
import json
import threading
import wget

#DataBaseDir = '/home/joe/project/workspace/python/kaggle/FashionFGVC5/data'
DataBaseDir = '../data'
success_download = set()
for i in range(32):
    SubDir = '%s/raw/images/train/%s' % (DataBaseDir, i)
    if(os.path.exists(SubDir) == True):
        ret = [int(im.split('.')[0]) for im in os.listdir(SubDir) if(im.endswith('.jpg'))]
        success_download.update(ret)

## load data
json_file = '%s/raw/%s.json' % (DataBaseDir, 'train')
with open(json_file, 'r') as i_file:
    json_data = json.load(i_file)
i_file.close()

## save images
ImageRootOutputDir = '%s/raw/images/%s' % (DataBaseDir, 'train')
if (os.path.exists(ImageRootOutputDir) == False):
    os.makedirs(ImageRootOutputDir)

json_data['images'] = [im_info for im_info in json_data['images'] if(im_info['imageId'] not in success_download)]
print('success %s, need to be downloaded %s' % (len(success_download), len(json_data['images'])))

class DownloadThread(threading.Thread):
    def __init__(self, t, low, high):
        super(DownloadThread, self).__init__()
        self.no = t
        self.low = low
        self.high = high
        ## save images
        self.ImageOutputDir = '%s/%s' % (ImageRootOutputDir, self.no)
        if (os.path.exists(self.ImageOutputDir) == False):
            os.makedirs(self.ImageOutputDir)

    def run(self):
            for i in range(self.low, self.high):
                try:
                    wget.download(json_data['images'][i]['url'],out='%s/%s.jpg' % (self.ImageOutputDir, json_data['images'][i]['imageId']))
                except:
                    continue

num_thread = 32
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