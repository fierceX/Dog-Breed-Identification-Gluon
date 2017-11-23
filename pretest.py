import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon import nn
from mxnet import nd
import pandas as pd
import mxnet as mx
import pickle
import numpy as np
from tqdm import tqdm
from model import get_net


data_dir = './data'
test_dir = 'test'
input_dir = 'train_valid_test'
valid_dir = 'valid'

netparams = '2'
csvname = 'p1.csv'
ids_synsets_name = 'ids_synsets'

input_str = data_dir + '/' + input_dir + '/'


f = open(ids_synsets_name,'rb')
ids_synsets = pickle.load(f)
f.close()

def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 299, 299)
    auglist = image.CreateAugmenter(data_shape=(3, 299, 299),
                        mean=np.array([0.485, 0.456, 0.406]),
                        std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

test_ds = vision.ImageFolderDataset(input_str + test_dir, flag=1,
                                     transform=transform_test)
valid_ds = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                     transform=transform_test)

loader = gluon.data.DataLoader

test_data = loader(test_ds, 64, shuffle=False, last_batch='keep')
valid_data = loader(valid_ds, 64, shuffle=True, last_batch='keep')

def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in tqdm(data):
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

def SaveTest(test_data,net,ctx,name,ids,synsets):
    outputs = []
    for data, label in tqdm(test_data):
        output = nd.softmax(net(data.as_in_context(ctx)))
        outputs.extend(output.asnumpy())
    with open(name, 'w') as f:
        f.write('id,' + ','.join(synsets) + '\n')
        for i, output in zip(ids, outputs):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')

net = get_net(netparams,mx.gpu())
net.hybridize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
print(get_loss(valid_data,net,mx.gpu()))

SaveTest(test_data,net,mx.gpu(),csvname,ids_synsets[0],ids_synsets[1])

