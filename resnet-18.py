import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms,models,utils
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

class Residual(nn.Module):
    # 初始化网络模型
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=(3, 3), padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=(3, 3), padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    # 前向传播
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        # 在通道维度上连结输出
        return F.relu(Y)


# # 定义组成ResNet的块
# 第1个块：使用（64个通道7×7卷积层） + （3×3最大池化层）
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                   )


# 第2~5个块，Residual
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):

    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# # 整合五个块，定义ResNet
model = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                      nn.Flatten(), nn.Linear(512, 10))

data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

if torch.cuda.is_available():
    model = model.cuda()
    # 定义损失函数和优化函数
criterion = nn.CrossEntropyLoss()  # 损失函数：损失函数交叉熵
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 优化函数：随机梯度下降法
epoch = 20
for i in range(epoch):
    losses = []
    for data in tqdm(train_loader):
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
        losses.append(loss.item())
        # 反向传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()
        optimizer.step()  # 更新参数
    print('*' * 10)
    print('epoch{}'.format(epoch))
    print('loss is {:.4f}'.format(sum(losses)/len(losses)))
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