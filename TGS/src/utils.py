import config
import time
import numpy as np
from contextlib import contextmanager

from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.color import rgb2gray

#from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
#import pydensecrf.densecrf as dcrf

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

def upsample(img):
    if config.img_size_original == config.img_size_target:
        return img
    return resize(img, (config.img_size_target, config.img_size_target), mode='constant', preserve_range=True)

def downsample(img):
    if config.img_size_original == config.img_size_target:
        return img
    return resize(img, (config.img_size_original, config.img_size_original), mode='constant', preserve_range=True)

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

"""
used for converting the decoded image to rle mask

"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)

# def crf(original_image, mask_img):
#     # Converting annotated image to RGB if it is Gray scale
#     if (len(mask_img.shape) < 3):
#         mask_img = gray2rgb(mask_img)
#
#     # #Converting the annotations RGB color to single 32 bit integer
#     annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
#
#     #     # Convert the 32bit integer color to 0,1, 2, ... labels.
#     colors, labels = np.unique(annotated_label, return_inverse=True)
#
#     n_labels = 2
#
#     # Setting up the CRF model
#     d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
#
#     # get unary potentials (neg log probability)
#     U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
#     d.setUnaryEnergy(U)
#
#     # This adds the color-independent term, features are the locations only.
#     d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
#                           normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#     # Run Inference for 10 steps
#     Q = d.inference(10)
#
#     # Find out the most probable class for each pixel.
#     MAP = np.argmax(Q, axis=0)
#
#     return MAP.reshape((original_image.shape[0], original_image.shape[1]))
