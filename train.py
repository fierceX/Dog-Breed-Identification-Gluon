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
from model import get_output

train_nd = nd.load('train_resnet152_v1.nd')

valid_nd = nd.load('valid_resnet152_v1.nd')

input_nd = nd.load('input_resnet152_v1.nd')

f = open('ids_synsets','rb')
ids_synsets = pickle.load(f)
f.close()

num_epochs = 100
batch_size = 128
learning_rate = 1e-4
weight_decay = 1e-5
lr_period = 40
lr_decay = 0.5
pngname='1'
modelparams='1'

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(train_nd[0],train_nd[1]), batch_size=batch_size,shuffle=True)
valid_data = gluon.data.DataLoader(gluon.data.ArrayDataset(valid_nd[0],valid_nd[1]), batch_size=batch_size,shuffle=True)
input_data = gluon.data.DataLoader(gluon.data.ArrayDataset(input_nd[0],input_nd[1]), batch_size=batch_size,shuffle=True)


def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, 
          lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    train_loss = []
    if valid_data is not None:
        test_loss = []
    
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        _loss = 0.
 #       if epoch > 0 and epoch % lr_period == 0:
 #           trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            _loss += nd.mean(loss).asscalar()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/len(train_data)
        train_loss.append(__loss)
        
        if valid_data is not None:  
            valid_loss = get_loss(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Valid loss %f, "
                         % (epoch,__loss , valid_loss))
            test_loss.append(valid_loss)
        else:
            epoch_str = ("Epoch %d. Train loss: %f, "
                         % (epoch, __loss))
            
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        

    plt.plot(train_loss, 'r')
    if valid_data is not None: 
        plt.plot(test_loss, 'g')
    plt.legend(['Train_Loss', 'Test_Loss'], loc=2)


    plt.savefig(pngname, dpi=1000)
    net.collect_params().save(modelparams)

ctx = mx.gpu()
net = get_output(ctx)
net.hybridize()

train(net, input_data,None, num_epochs, learning_rate, weight_decay, 
      ctx, lr_period, lr_decay)
