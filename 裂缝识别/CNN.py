import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms,models,utils
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # 按照公式计算后经过卷积层不改变尺寸
        self.pool = nn.MaxPool2d(2, 2)  # 2*2的池化 池化后size 减半
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 256)  # 两个池化，所以是224/2/2=56
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    #         self.dp = nn.Dropout(p=0.5)
    def forward(self, x):
        #         print("input:", x)
        x = self.pool(F.relu(self.conv1(x)))
        #         print("first conv:", x)
        x = self.pool(F.relu(self.conv2(x)))
        #         print("second conv:", x)

        x = x.view(-1, 16 * 56 * 56)  # 将数据平整为一维的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

data_transform = transforms.Compose([
    transforms.Resize(size = (224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])

batch_size = 16
learning_rate = 1e-2
num_epoches = 20

dataset = ImageFolder('D:/pycharm/data/archive', transform=data_transform)
full_ds = dataset
train_size = int(0.8 * len(full_ds))
validate_size = len(full_ds) - train_size
train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=True)

model = CNN()
if torch.cuda.is_available():
    model = model.cuda()
    # 定义损失函数和优化函数
criterion = nn.CrossEntropyLoss()  # 损失函数：损失函数交叉熵
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 优化函数：随机梯度下降法
epoch = 0
for data in train_loader:
    img, label = data
    img = Variable(img)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    # 前向传播
    out = model(img)
    loss = criterion(out, label)
    # 反向传播
    optimizer.zero_grad()  # 梯度归零
    loss.backward()
    optimizer.step()  # 更新参数
    epoch += 1
    if (epoch) % 10 == 0:
        print('*' * 10)
        print('epoch{}'.format(epoch))
        print('loss is {:.4f}'.format(loss.item()))
    # 测试网络
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / (len(test_ds)), eval_acc / (len(test_ds))))