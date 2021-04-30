import torch
import torchvision.transforms as transforms
from PIL import Image
from config.config import model_dict
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--model', nargs='?', type=str, default='lenet')
args = parser.parse_args()

assert (args.model in model_dict.keys()), "only support {}".format(
    model_dict.keys())

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

device = None
# get device
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

net = model_dict[args.model]["model"](False, cfg=model_dict[args.model])
net = net.to(device)
model_info(net)