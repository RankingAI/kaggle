import tensorflow as tf
import keras.backend as K
from pathlib import Path
from scipy import ndimage
import imageio
import os,sys
import numpy as np

import config

def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(true,pred):
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return K.sum(intersection, axis=-1) / (K.sum(union, axis=-1) + K.epsilon())

def competitionMetric2(true, pred): #any shape can go

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

if __name__ == '__main__':
    # Get image
    im_id = '2bf5343f03'
    im_dir = Path('%s/raw/train' % config.DataBaseDir)
    im_path = im_dir / 'images' / '{}.png'.format(im_id)
    img = imageio.imread(im_path.as_posix())

    # Get mask
    im_path = im_dir / 'masks' / '{}.png'.format(im_id)
    target_mask = imageio.imread(im_path.as_posix())
    print(target_mask[:5])

    # Fake prediction mask
    pred_mask = np.load('%s/unet_depth/predict/masks/%s.npy' % (config.ModelRootDir, im_id))
    print(pred_mask[:5])

    ret = competitionMetric2(target_mask, pred_mask)
    print(ret)
