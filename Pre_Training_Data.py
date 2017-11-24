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
from model import get_features,transform_test,transform_train

data_dir = '../data/kaggle_dog'
train_dir = 'train'
test_dir = 'test'
valid_dir = 'valid'
input_dir = 'train_valid_test'

train_valid_dir = 'train_valid'

input_str = data_dir + '/' + input_dir + '/'


batch_size = 128

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

net = get_features(mx.gpu())
net.hybridize()


def SaveNd(data,net,name):
    x =[]
    y =[]
    print('提取特征 %s' % name)
    for fear1,fear2,label in tqdm(data):
        fear1 = fear1.as_in_context(mx.gpu())
        fear2 = fear2.as_in_context(mx.gpu())
        out = net(fear1,fear2).as_in_context(mx.cpu())
        x.append(out)
        y.append(label)
    x = nd.concat(*x,dim=0)
    y = nd.concat(*y,dim=0)
    print('保存特征 %s' % name)
    nd.save(name,[x,y])


SaveNd(train_data,net,'train_inception_v3.nd')
SaveNd(valid_data,net,'valid_inception_v3.nd')
SaveNd(train_valid_data,net,'input_inception_v3.nd')
# SaveNd(test_data,net,'test_resnet152_v1.nd')
ids = ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
synsets = train_valid_ds.synsets
f = open('ids_synsets','wb')
pickle.dump([ids,synsets],f)
f.close()