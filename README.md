# Dog-Breed-Identification-Gluon
[Kaggle 120种狗分类比赛](https://www.kaggle.com/c/dog-breed-identification) Gluon实现代码  
通过[Gluon教程](http://zh.gluon.ai/chapter_computer-vision/kaggle-gluon-dog.html)，边学边练。使用`renset152_v1`和`inception_v3`进行模型融合得到`0.27760`分数

详细说明参见[Gluon炼丹（Kaggle 120种狗分类，迁移学习加双模型融合）](https://fiercex.github.io/post/gluon_kaggle/)

在[杨培文](https://github.com/ypwhs/DogBreed_gluon)的代码中学到的，如果使用imagenet网络进行预训练，锁住特征层的话，那可以先把所有数据都都过一边特征层网络，这样在后面进行分类输出网络的训练时会省去很多时间和显存，并且在训练的时候可以把bs开到很大，而且在过特征层网络的时候，可以把bs设的小一点，而且不用反向传播，速度也不会太慢，这样就解决了在低配电脑上无法训练的问题。而且用最新版mxnet输出的特征数据非常小，70M左右，后面的分类层即使用cpu也能跑很顺畅
