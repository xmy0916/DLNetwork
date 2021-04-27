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
from model.LeNet import LeNet

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--model', nargs='?', type=str, default='lenet')
args = parser.parse_args()

model_dict = {
    "lenet": {
        "model": LeNet,
        "save_path": "params/lenet.pth"
    }
}

assert (args.model in model_dict.keys()), "only support {}".format(
    model_dict.keys())

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=False, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=0)

test_data_iter = iter(test_loader)
test_images, test_labels = test_data_iter.next()

net = model_dict[args.model]['model']()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印信息
        running_loss += loss.item()
        if step % 500 == 499:  # 每500个batch打印一次训练状态
            with torch.no_grad():
                outputs = net(test_images)
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_labels).sum().item() / test_labels.size(0)
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), model_dict[args.model]['save_path'])