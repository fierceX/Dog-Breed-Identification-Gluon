from mxnet import gluon
from mxnet import init
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn

def get_features(ctx):
    resnet = vision.inception_v3(pretrained=True,ctx=ctx)
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