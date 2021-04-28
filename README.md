# 深度学习基础网络实现
系列文章地址：[深度学习基础网络实现-csdn](https://blog.csdn.net/qq_37668436/article/details/116199117) 

框架：pytorch1.8.1

python：python3.7

# 网络
- [x] [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) 
- [x] [AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) 
- [ ] [VGG](https://arxiv.org/pdf/1409.1556.pdf) 
- [ ] [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf) 
- [ ] [ResNet](https://arxiv.org/pdf/1512.03385.pdf) 

# 细节
|模型|LeNet|AlexNet|
| :---: | :---: | :---: |
|轮数|5|5|
|精度|0.622|0.800、0.860(加pretrain)|
|日志|[lenet.log](./logs/lenet.log)|[alexnet.log](./logs/alexnet.log)、[alexnet_pretrain.log](./logs/alexnet_pretrain.log)|


# 参数
- model: 选择使用的模型（字符串类型，例如：--model lenet）
- save_feature: 设置是否保存特征图(布尔类型，例如：--save_feature True）

# 训练
```bash
python3 train.py --model lenet
```
# 测试
```bash
python3 predict.py --model lenet
```

