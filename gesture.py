from __future__ import print_function
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

from GestureDataset import GestureDataset

train_dataset = GestureDataset(root=osp.abspath('./gesture_file'), train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
img, target = train_dataset.__getitem__(1)

test_dataset = GestureDataset(root=osp.abspath('./gesture_file'), train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # self.fc = nn.Linear(3380, 6)
        self.fc1 = nn.Linear(3380, 600)
        self.fc2 = nn.Linear(600, 6)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        # x = self.fc(x)
        # return F.log_softmax(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure STN the performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor, nrow=4))
        out_grid = convert_image_np(torchvision.utils.make_grid(input_tensor, nrow=4))

        # Plot the results side-by-side
        f, axarr = plt.subplots()
        axarr.imshow(torchvision.utils.make_grid(input_tensor, nrow=4))
        axarr.set_title('Dataset Images')

        #axarr[1].imshow(out_grid)
        #axarr[1].set_title('Transformed Images')


if __name__ == '__main__':
    for epoch in range(1, 2):
        train(epoch)
        test()

    visualize_stn()
    plt.ioff()
    plt.show()

