DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

debug = False
depth_channel = False

kfold = 10
kfold_seed = 2018
stratified = True

img_size_original = 101
infer_batch_size = 1024 

stages = {
    'unet_res_block': 2,
    'unet_resnet_v2': 1,
    'unet_vgg16': 1,
    'unet_resnet50_vgg16': 2,
    'unet_xception': 2,
    'deeplab_v3': 1,
}

encoder_input_size = {
    'unet_res_block': 128,
    'unet_resnet_v2': 256,
    'unet_vgg16': 128,
    'unet_resnet50_vgg16': 256,
    'unet_xception': img_size_original,
    'deeplab_v3': 128,
}

batch_size = {
    'unet_res_block': 16,
    'unet_resnet_v2': 16,
    'unet_vgg16': 32,
    'unet_resnet50_vgg16': 32,
    'unet_xception': 16,
    'deeplab_v3': 16,
}

epochs = {
    'unet_res_block': [60, 60],
    'unet_resnet_v2': [100, 20],
    'unet_vgg16': [60],
    'unet_resnet50_vgg16': [60, 60],
    'unet_xception': [60, 60],
    'deeplab_v3': [80],
}

learning_rate = {
    'unet_res_block': [0.02, 0.01],
    'unet_resnet_v2': [0.001, 0.0001],
    'unet_vgg16': 0.0005,
    'unet_resnet50_vgg16': [0.001, 0.0005],
    'unet_xception': [0.001, 0.0005],
    'deeplab_v3': [0.0005],
}

freeze_till_layer = {
    'unet_res_block': None,
    'unet_resnet_v2': 'input_1',
    'unet_vgg16': None,
    'unet_resnet50_vgg16': 'input_1',
    'unet_xception': 'input_1',
    'deeplab_v3': 'average_pooling2d_1',
}

grayscale = {
    'unet_res_block': True,
    'unet_resnet_v2': False,
    'unet_vgg16': False,
    'unet_resnet50_vgg16': False,
    'unet_xception': False,
    'deeplab_v3': False,
}
