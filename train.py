# 为了下载数据集，取消全局验证
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch import optim
from config.config import model_dict
import os

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--save_feature', nargs='?', type=bool, default=False)
parser.add_argument('--model', nargs='?', type=str, default='lenet')
args = parser.parse_args()

assert (args.model in model_dict.keys()), "only support {}".format(
    model_dict.keys())
train_bs, test_bs = model_dict[args.model]["batch_size"]
transform = transforms.Compose(
    [transforms.Resize(model_dict[args.model]["input_size"]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, shuffle=False, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=0)

test_data_iter = iter(test_loader)
test_images, test_labels = test_data_iter.next()

net = model_dict[args.model]['model'](saveFeature=args.save_feature, cfg=model_dict[args.model])

device = None
# get device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
net = net.to(device)
print(net)
print(model_dict[args.model])
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=model_dict[args.model]['lr'])

for epoch in range(model_dict[args.model]['epoch']):
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印信息
        running_loss += loss.item()
        print_step = 100
        if step % print_step == 0:  # 每500个batch打印一次训练状态
            with torch.no_grad():
                test_images = test_images.to(device)
                # text_labels = test_labels.to(device)
                outputs = net(test_images)
                predict_y = torch.max(outputs, dim=1)[1].cpu()
                text_labels = test_labels.cpu()

                accuracy = (predict_y == test_labels).sum().item() / test_labels.size(0)
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / print_step, accuracy))
                running_loss = 0.0

print('Finished Training')
index = model_dict[args.model]['save_path'].rfind("/")
farther_path = model_dict[args.model]['save_path'][:index]
if not os.path.exists(farther_path):
    os.makedirs(farther_path)
torch.save(net.state_dict(), model_dict[args.model]['save_path'])
