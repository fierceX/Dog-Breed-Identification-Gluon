from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet import nd
import numpy as np
import mxnet as mx
import pandas as pd
import pickle
from tqdm import tqdm
import os

data_dir = '../data/kaggle_dog'
train_dir = 'train'
test_dir = 'test'
valid_dir = 'valid'
input_dir = 'train_valid_test'

train_valid_dir = 'train_valid'

input_str = data_dir + '/' + input_dir + '/'


batch_size = 256

def transform_train(data, label):
    im = image.imresize(data.astype('float32') / 255, 224, 224)
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 224, 224)
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224),
                        mean=np.array([0.485, 0.456, 0.406]),
                        std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

train_ds = vision.ImageFolderDataset(input_str + train_dir, flag=1,
                                     transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + train_valid_dir,
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + test_dir, flag=1,
                                     transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True,
                          last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

def getnet(ctx):
    resnet = gluon.model_zoo.vision.resnet152_v1(pretrained=True,ctx=ctx)
    return resnet.features

net = getnet(mx.gpu())
net.hybridize()

def SaveNd(data,net,name):
    x =[]
    y =[]
    print('提取特征 %s' % name)
    for fear,label in tqdm(data):
        x.append(net(fear.as_in_context(mx.gpu())).as_in_context(mx.cpu()))
        y.append(label)
    x = nd.concat(*x,dim=0)
    y = nd.concat(*y,dim=0)
    print('保存特征 %s' % name)
    nd.save(name,[x,y])


SaveNd(train_data,net,'train_resnet152_v1.nd')
SaveNd(valid_data,net,'valid_resnet152_v1.nd')
SaveNd(train_valid_data,net,'input_resnet152_v1.nd')
# SaveNd(test_data,net,'test_resnet152_v1.nd')
ids = ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
synsets = train_valid_ds.synsets
f = open('ids_synsets','wb')
pickle.dump([ids,synsets],f)
f.close()