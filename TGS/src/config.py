DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

debug = False 

kfold = 5
kfold_seed = 2018
stratified = True

img_size_original = 101
infer_batch_size = 500

stages = {
    'unet_res_block': 2,
    'unet_resnet_v2': 2,
    'unet_vgg16': 1,
    'unet_resnet50_vgg16': 1,
}

encoder_input_size = {
    'unet_res_block': img_size_original,
    'unet_resnet_v2': 256,
    'unet_vgg16': 128,
    'unet_resnet50_vgg16': 256,
}

batch_size = {
    'unet_res_block': 64,
    'unet_resnet_v2': 16,
    'unet_vgg16': 32,
    'unet_resnet50_vgg16': 32,
}

epochs = {
    'unet_res_block': [35, 40],
    'unet_resnet_v2': [25, 25],
    'unet_vgg16': [60],
    'unet_resnet50_vgg16': [20, 20],
}

learning_rate = {
    'unet_res_block': 0.001,
    'unet_resnet_v2': 0.0005,
    'unet_vgg16': 0.005,
    'unet_resnet50_vgg16': 0.00025,
}

freeze_till_layer = {
    'unet_res_block': None,
    'unet_resnet_v2': 'input_1',
    'unet_vgg16': None,
    'unet_resnet50_vgg16': 'input_1',
}

grayscale = {
    'unet_res_block': True,
    'unet_resnet_v2': False,
    'unet_vgg16': False,
    'unet_resnet50_vgg16': False,
}
