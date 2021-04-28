from model.LeNet import LeNet
from model.AlexNet import AlexNet,AlexNet_pretrain

model_dict = {
    "lenet": {
        "model": LeNet,
        "epoch":5,
        "num_class":10,
        "save_path": "params/lenet.pth",
        "input_size":(32,32),
        "batch_size":(36,1000),
        "lr":0.001
    },
    "alexnet":{
        "model":AlexNet, # 加载预训练模型的alexnet写AlexNet_pretrain
        "epoch":5,
        "num_class":10,
        "save_path": "params/alexnet.pth",
        "input_size":(224,224),
        "batch_size":(8,1000),
        "lr":0.0001
    }
}
