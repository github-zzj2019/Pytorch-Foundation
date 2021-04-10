import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import  Variable



Batch_Size=4


#数据集

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size,shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size,shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#定义模型VGG16
class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2,2))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2,2))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 1, padding=(0, 0))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2),stride=(2,2))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 1, padding=(0, 0))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2),stride=(2,2))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 1, padding=(0, 0))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), stride=(2,2))  # pooling 512 * 7 * 7

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(-1, 4)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out

# our model
net=VGG16()
##print(net)
# Define loss (Cross-Entropy)
criterion = nn.CrossEntropyLoss()
# SGD with momentum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ ==  '__main__':
    # Train the network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data

            # warp them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")

    print("Beginning Testing")
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))