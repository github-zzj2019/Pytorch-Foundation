import torchvision as tv
import torch as t
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

show = ToPILImage()

EPOCH = 1
LR = 0.01
MOMTONTU = 0.9
BATCH_SIZE=4

transform = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = tv.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0)

testset = tv.datasets.CIFAR10(
    './data',
    train=False,
    download=True,
    transform=transform)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
print(classes[label])

show((data + 1) / 2).resize((100, 100))

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(' '.join('%11s' % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100))

import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            # 1-1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 1-2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 1-3
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            # 2-1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 2-2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 2-3
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            # 3-1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 3-2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 3-3
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 3-4
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            # 4-1
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 4-2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 4-3
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 4-4
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            # 5-1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 5-2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 5-3
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 5-4
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.layer7 = nn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.layer8 = nn.Sequential(

            # 8 output layer
            nn.Linear(4096, 10),
            nn.BatchNorm1d(10),
            nn.Softmax())

    def forward(self, x):
        # print('layer 0:',x.size())
        out = self.layer1(x)
        # print('layer 1:',out.size())
        out = self.layer2(out)
        # print('layer 2:',out.size())
        out = self.layer3(out)
        # print('layer 3:',out.size())
        out = self.layer4(out)
        # print('layer 4:',out.size())
        out = self.layer5(out)
        # print('layer 5:',out.size())
        vgg16_features = out.view(out.size()[0], -1)
        # print('layer 5-6:',vgg16_features.size())
        out = self.layer6(vgg16_features)
        # print('layer 6:',out.size())
        out = self.layer7(out)
        # print('layer 7:',out.size())
        out = self.layer8(out)
        # print('layer 85:',out.size())

        return out


net = Net()#.cuda()
# print(net)

# test model
print('input_size', images.size())
print('input_size', labels.size())

from torch import optim

criterion = t.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMTONTU)

for epoch in range(EPOCH):

    running_loss = 0.0
    for i, data in enumerate(trainloader):

        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()  # 一个batch返回4张图片
print('实际的label: ', ' '.join( \
    '%08s' % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))

# Test the model
correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数

for i, data in enumerate(testloader):
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))