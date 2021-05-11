from model.LeNet import LeNet
from model.AlexNet import AlexNet, AlexNet_pretrain
from model.VggNet import VggNet
from model.GoogLeNet import googlenet
model_dict = {
    "lenet": {
        "model": LeNet,
        "epoch": 5,
        "num_class": 10,
        "save_path": "params/lenet.pth",
        "input_size": (32, 32),
        "batch_size": (36, 1000),
        "lr": 0.001
    },
    "alexnet": {
        "model": AlexNet,  # 加载预训练模型的alexnet写AlexNet_pretrain
        "epoch": 5,
        "num_class": 10,
        "save_path": "params/alexnet.pth",
        "input_size": (224, 224),
        "batch_size": (8, 1000),
        "lr": 0.0001
    },
    "vggnet": {
        "model": VggNet,
        # name support list:[
        #     'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
        #     'vgg19_bn', 'vgg19'
        #     ]
        "name": "vgg16",
        "pretrained": True,
        "epoch": 5,
        "num_class": 10,
        "save_path": "params/vgg.pth",
        "input_size": (112, 112),  # 224跑不动...
        "batch_size": (8, 1000),
        "lr": 0.0001
    },
    "googlenet": {
        "model": googlenet,
        "pretrained": True,
        "epoch": 5,
        "num_class": 10,
        "save_path": "params/googlenet.pth",
        "input_size": (112, 112),  # 224跑不动...
        "batch_size": (8, 1000),
        "lr": 0.0001
    },
    
}
