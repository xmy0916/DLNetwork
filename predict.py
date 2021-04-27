import torch
import torchvision.transforms as transforms
from PIL import Image
from model.LeNet import LeNet
from config.config import model_dict
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--model', nargs='?', type=str, default='lenet')
parser.add_argument('--save_feature', nargs='?', type=bool, default=False)
args = parser.parse_args()


assert (args.model in model_dict.keys()), "only support {}".format(
    model_dict.keys())

transform = transforms.Compose(
    [transforms.Resize(model_dict[args.model]["input_size"]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = model_dict[args.model]["model"](args.save_feature)
net.load_state_dict(torch.load(model_dict[args.model]["save_path"]))

img = Image.open("data/ship.png").convert('RGB')
img = transform(img)
img = torch.unsqueeze(img, dim=0)

with torch.no_grad():
    output = net(img)
    predict = torch.max(output, dim=1)[1].data.numpy()
print(classes[int(predict)])
