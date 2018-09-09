DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

kfold = 5
kfold_seed = 2018
stratified = True

img_size_original = 101
img_size_target = 128

debug = False

# u-net super params
#strategy = 'unet_depth'
strategy = 'unet_res_block'
epochs = [50, 50]
batch_size = 32
