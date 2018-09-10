DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

debug = False

kfold = 5
kfold_seed = 2018
stratified = True

img_size_original = 101

encoder_input_size = {
    'unet_res_block': img_size_original,
    'unet_resnet_v2': 256,
    'unet_vgg16': 128,
}

batch_size = {
    'unet_res_block': 64,
    'unet_resnet_v2': 16,
    'unet_vgg16': 32,
}

epochs = {
    'unet_res_block': [35, 40],
    'unet_resnet_v2': [25, 25],
    'unet_vgg16': [40],
}

learning_rate = {
    'unet_res_block': 0.001,
    'unet_resnet_v2': 0.0025,
    'unet_vgg16': 0.005,
}

freeze_till_layer = {
    'unet_res_block': None,
    'unet_resnet_v2': 'input_1',
    'unet_vgg16': None,
}

stages = {
    'unet_res_block': 2,
    'unet_resnet_v2': 2,
    'unet_vgg16': 1,
}

grayscale = {
    'unet_res_block': True,
    'unet_resnet_v2': False,
    'unet_vgg16': False,
}

