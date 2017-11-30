import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data import vision
from mxnet import nd
import mxnet as mx
import pickle
from tqdm import tqdm
from model import get_net,transform_test


batch_size = 128

data_dir = './data'
test_dir = 'test'
input_dir = 'train_valid_test'
valid_dir = 'valid'

netparams = 'train.params'
csvname = 'kaggle.csv'
ids_synsets_name = 'ids_synsets'

input_str = data_dir + '/' + input_dir + '/'


f = open(ids_synsets_name,'rb')
ids_synsets = pickle.load(f)
f.close()


test_ds = vision.ImageFolderDataset(input_str + test_dir, flag=1,
                                     transform=transform_test)
valid_ds = vision.ImageFolderDataset(input_str + valid_dir, flag=1,
                                     transform=transform_test)

loader = gluon.data.DataLoader

test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')

def get_loss(data, net, ctx):
    loss = 0.0
    for feas1,feas2, label in tqdm(data):
        label = label.as_in_context(ctx)
        feas1 = feas1.as_in_context(ctx)
        feas2 = feas2.as_in_context(ctx)
        output = net(feas1,feas2)
        cross_entropy = softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

def SaveTest(test_data,net,ctx,name,ids,synsets):
    outputs = []
    for data1,data2, label in tqdm(test_data):
        data1 =data1.as_in_context(ctx)
        data2 =data2.as_in_context(ctx)
        output = nd.softmax(net(data1,data2))
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

