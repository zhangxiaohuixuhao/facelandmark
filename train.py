import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.optim as optim
from torch.autograd import Variable
from model import *


lr = 0.01
momentum = 0.5
epochs =50
batch_size = 64
test_batch_size = 1000

def train(epoch, train_loader):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) #交叉熵损失
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                       100. * batch_id / len(train_loader), loss.item()))

def test(epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target)
        # output.shape-->[64, 10] 取每一列的最大值
        # output.data.max(1)取每一列的最大值，返回值有两个元素
        # 第一个元素是每个最大值的值，第二个元素是每个最大值所在的行数
        pred = output.data.max(1, keepdim=True)[1] 
        # view 是resize的最用
        # view_as：a = torch.Tensor(2, 4)
        #          tmp = torch.Tensor(2, 4)
        #          b = a.view_as(tmp)
        #          b.data.shape 应该是（4，2）即输出与传入参数tmp的大小相同
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = torch.utils.data.DataLoader( #加载训练数据
                datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader( #加载训练数据
                datasets.MNIST('./data', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=batch_size, shuffle=True
    )
    model = LeNet(num_classes=10)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        train(epoch, train_loader)
        test(epoch, test_loader)
    torch.save(model, 'model.pth')