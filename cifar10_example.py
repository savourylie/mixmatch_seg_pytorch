import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms

from data import CIFAR10

DATA_ROOT = './data/cifar/'

BATCH_SIZE = 32
EPOCHS = 10
LOG_INTERVAL = 100


def test(model, test_loader):
    print("Testing...\n")

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    num_rows = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            if torch.cuda.is_available():
                data, target = data.cuda(async=True), target.cuda(async=True)
            else:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target).unsqueeze(1).float()
            num_rows += data.size(0)

            logit = model(data)
            test_loss += criterion(logit, target)

            pred = torch.argmax(output, dim=1)
            correct += (pred == target).cpu().sum()

        test_loss /= num_rows

        
        print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
               test_loss, correct, num_rows, (100. * correct.item()) / num_rows))

    return (100. * correct.item()) / num_rows


if __name__ == '__main__':

    transform_train = transforms.Compose([
                                              # transforms.CenterCrop(160),
                                              # transforms.RandomCrop(160, padding=20),
                                              # transforms.RandomHorizontalFlip(),
                                              # transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              # transforms.Normalize((0.7089, 0.5047, 0.6507), (0.1280, 0.1978, 0.1358)),
                                              # Cutout(n_holes=4, length=10)
                                             ])

    transform_test = transforms.Compose([
                                         # transforms.CenterCrop(160),
                                         transforms.ToTensor(),
                                         # transforms.Normalize((0.7089, 0.5047, 0.6507), (0.1280, 0.1978, 0.1358))
                                        ])

    training_set = CIFAR10(root=DATA_ROOT + 'train/', resize=224, transform=transform_train)
    test_set = CIFAR10(root=DATA_ROOT + 'test/', resize=224, transform=transform_test)

    kwargs = {'num_workers': 4 * len(GPU_IDS.split(',')), 'pin_memory': False} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, **kwargs)

    model = models.resnet50()
    model.fc = nn.Linear(2048, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()


    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()

        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(async=True), target.cuda(async=True)
            else:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target).unsqueeze(1).float()

            optimizer.zero_grad()
            logit = model(data)
            loss_mean = criterion(logit, target)

            optimizer.step()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Training loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), train_loader.dataset.len,
                    100. * batch_idx * len(data) / train_loader.dataset.len, loss_mean.data.item()))

        print("Epoch {} took {} seconds.".format(epoch, time.time() - epoch_start_time))

        print("=====================")
        acc = test(model, test_loader)
        print("=====================")

    print("Training took {} seconds\n".format(round(time.time() - training_start_time, 2)))