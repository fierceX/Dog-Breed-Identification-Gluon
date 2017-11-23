# Dog-Breed-Identification-Gluon
[Kaggle 120种狗分类比赛](https://www.kaggle.com/c/dog-breed-identification) Gluon实现代码  
通过[Gluon教程](http://zh.gluon.ai/chapter_computer-vision/kaggle-gluon-dog.html)，边学边练。使用renset152_v1得到0.37900分数

## 复现
- 在Kaggle上下载训练数据和测试数据，在`reorg_dog_data`设置相关路径并运行，整理数据集
- 在`model`中设置好预训练网络和输出网络
- 运行`Pre_Training_Data`生成预训练数据
- 在`train`中调整参数并运行进行训练
- 运行`pretest`生成结果csv文件，提交到kaggle

## 原理
在[杨培文](https://github.com/ypwhs/DogBreed_gluon)的代码中学到的，如果使用imagenet网络进行预训练，锁住特征层的话，那可以先把所有数据都都过一边特征层网络，这样在后面进行分类输出网络的训练时会省去很多时间和显存，并且在训练的时候可以把bs开到很大，而且在过特征层网络的时候，可以把bs设的小一点，而且不用反向传播，速度也不会太慢，这样就解决了在低配电脑上无法训练的问题。而且用最新版mxnet输出的特征数据非常小，70M左右，后面的分类层即使用cpu也能跑很顺畅
