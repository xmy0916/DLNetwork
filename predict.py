import torch
import torchvision.transforms as transforms
from PIL import Image
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

device = None
# get device
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

net = model_dict[args.model]["model"](args.save_feature, cfg=model_dict[args.model])
net.load_state_dict(torch.load(model_dict[args.model]["save_path"]))
net = net.to(device)

img = Image.open("data/ship.png").convert('RGB')
img = transform(img)
img = torch.unsqueeze(img, dim=0)
img = img.to(device)

with torch.no_grad():
    output = net(img).cpu()
    predict = torch.max(output, dim=1)[1].data.numpy()
print(classes[int(predict)])
