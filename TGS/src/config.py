DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

kfold = 8
kfold_seed = 2018
stratified = True

img_size_original = 101
img_size_target = 128

# u-net super params
strategy = 'unet'
epochs = 200
batch_size = 64
