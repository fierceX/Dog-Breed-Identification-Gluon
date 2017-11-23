from mxnet import gluon
from mxnet import init
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
from mxnet import image
import numpy as np
from mxnet import nd

def get_features2(ctx):
    resnet = vision.inception_v3(pretrained=True,ctx=ctx)
    return resnet.features

def get_features1(ctx):
    resnet = vision.resnet152_v1(pretrained=True,ctx=ctx)
    return resnet.features

def get_output(ctx,ParamsName=None):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dropout(.7))
        net.add(nn.Dense(120))
    if ParamsName is not None:
        net.collect_params().load(ParamsName,ctx)
    else:
        net.initialize(init = init.Xavier(),ctx=ctx)
    return net

def get_net(ParamsName,ctx):
    output = get_output(ctx,ParamsName)
    features = get_features(ctx)
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(features)
        net.add(output)
    return net

def transform_train(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist1:
        im1 = aug(im1)
    for aug in auglist2:
        im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2,0,1))
    im2 = nd.transpose(im2, (2,0,1))
    return (im1,im2, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im1 = image.imresize(data.astype('float32') / 255, 224, 224)
    im2 = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist1 = image.CreateAugmenter(data_shape=(3, 224, 224),
                        mean=np.array([0.485, 0.456, 0.406]), 
                        std=np.array([0.229, 0.224, 0.225]))
    auglist2 = image.CreateAugmenter(data_shape=(3, 299, 299),
                        mean=np.array([0.485, 0.456, 0.406]), 
                        std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist1:
        im1 = aug(im1)
    for aug in auglist2:
        im2 = aug(im2)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im1 = nd.transpose(im1, (2,0,1))
    im2 = nd.transpose(im2, (2,0,1))
    return (im1,im2, nd.array([label]).asscalar().astype('float32'))