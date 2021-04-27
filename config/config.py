from model.LeNet import LeNet
from model.AlexNet import AlexNet,AlexNet_pretrain

model_dict = {
    "lenet": {
        "model": LeNet,
        "save_path": "params/lenet.pth",
        "input_size":(32,32),
        "batch_size":(36,1000)
    },
    "alexnet":{
        "model":AlexNet_pretrain, # 加载预训练模型的alexnet写AlexNet_pretrain
        "num_class":10,
        "save_path": "params/alexnet.pth",
        "input_size":(227,227),
        "batch_size":(8,1000)
    }
}