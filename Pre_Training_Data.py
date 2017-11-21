from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet import nd
import numpy as np
import mxnet as mx
import pandas as pd

data_dir = './'


train_rec = 'train.rec'
input_rec = 'train_valid.rec'
valid_rec = 'valid.rec'

bs = 256

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


# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageRecordDataset(data_dir + train_rec, flag=1, 
                                     transform=transform_train)
valid_ds = vision.ImageRecordDataset(data_dir + valid_rec, flag=1, 
                                     transform=transform_test)
train_valid_ds = vision.ImageRecordDataset(data_dir + input_rec, 
                                           flag=1, transform=transform_train)

df_test = pd.read_csv('sample_submission.csv')
n_test = len(df_test)
Xtest = nd.zeros((n_test, 3, 224, 224))

for i, fname in enumerate(df_test['id']):
    img = image.imread('../data/kaggle_dog/train_valid_test/test/unknown/%s.jpg' % fname)
    img = image.imresize(img.astype('float32') / 255, 224, 224)
    
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224),
                        mean=np.array([0.485, 0.456, 0.406]),
                        std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        img = aug(img)
    img = nd.transpose(img, (2,0,1))
    Xtest[i] = img
    nd.waitall()

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size=bs, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size=bs, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size=bs, shuffle=True, 
                          last_batch='keep')

test_data = loader(Xtest,batch_size=bs, shuffle=True, last_batch='keep')

trainX = nd.zeros((len(train_ds),2048,7,7),ctx = mx.cpu())
validX = nd.zeros((len(valid_ds),2048,7,7),ctx = mx.cpu())
inputX = nd.zeros((len(train_valid_ds),2048,7,7),ctx = mx.cpu())
testX = nd.zeros((len(Xtest),2048,7,7),ctx=mx.cpu())

trainy = nd.zeros((len(train_ds)),ctx = mx.cpu())
validy = nd.zeros((len(valid_ds)),ctx = mx.cpu())
inputy = nd.zeros((len(train_valid_ds)),ctx = mx.cpu())

def getnet(ctx):
    resnet = gluon.model_zoo.vision.resnet152_v1(pretrained=True,ctx=ctx)
    return resnet.features

net = getnet(mx.gpu())
net.hybridize()

def SaveNd(data,net,xname,yname,x,y,bs):
    for i,(fear,label) in enumerate(data):
        n = len(fear)
        print(n)
        if n == bs:
            x[i*bs:(i+1)*bs] = net(fear.as_in_context(mx.gpu())).as_in_context(mx.cpu())
            y[i*bs:(i+1)*bs] = label
        else:
            x[i*bs:] = net(fear.as_in_context(mx.gpu())).as_in_context(mx.cpu())
            y[i*bs:] = label
    nd.save(xname,x)
    nd.save(yname,y)

def SaveTest(data,net,xname,x,bs):
    for i,fear in enumerate(data):
        n = len(fear)
        print(n)
        if n == bs:
            x[i*bs:(i+1)*bs] = net(fear.as_in_context(mx.gpu())).as_in_context(mx.cpu())
        else:
            x[i*bs:] = net(fear.as_in_context(mx.gpu())).as_in_context(mx.cpu())
    nd.save(xname,x)

SaveNd(train_data,net,'train_resnet152_v1_x.nd','train_resnet152_v1_y.nd',trainX,trainy,bs)
SaveNd(valid_data,net,'valid_resnet152_v1_x.nd','valid_resnet152_v1_y.nd',validX,validy,bs)
SaveNd(train_valid_data,net,'input_resnet152_v1_x.nd','input_resnet152_v1_y.nd',inputX,inputy,bs)
SaveTest(test_data,net,'test_resnet152_v1_x.nd',testX,bs)