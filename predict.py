import torch
import torchvision.transforms as transforms
from PIL import Image
from model.LeNet import LeNet
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--model', nargs='?', type=str, default='lenet')
args = parser.parse_args()

model_dict = {"lenet": {"model":LeNet,"save_path":"params/lenet.pth"}}

assert (args.model in model_dict.keys()), "only support {}".format(
    model_dict.keys())

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = model_dict[args.model]["model"]()
net.load_state_dict(torch.load(model_dict[args.model]["save_path"]))

img = Image.open("data/ship.png").convert('RGB')
img = transform(img)
img = torch.unsqueeze(img, dim=0)

with torch.no_grad():
    output = net(img)
    predict = torch.max(output, dim=1)[1].data.numpy()
print(classes[int(predict)])
