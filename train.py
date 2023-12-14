'''
This is to finetune a model for the CUB-200 dataset
'''

import argparse
parser = argparse.ArgumentParser(description = "Template")

parser.add_argument("-gpu", "--GPU_index", default = 0, type = int, help = "gpu index")
parser.add_argument("-bs", "--batch_size", default = 128, type = int, help = "batch size")
parser.add_argument("-lr", "--learning_rate", default = 1e-3, type = float, help = "learning rate")
parser.add_argument("-mt", "--momentum", default = 0.9, type = float, help = "momentum")
parser.add_argument("-wd", "--weight_decay", default = 5e-4, type = float, help = "weight decay")
parser.add_argument("-n", "--n_epoch", default = 200, type = int, help = 'number of epochs')
parser.add_argument("-model", "--model_name", default = 'vgg', type = str, help = "the name of the model")
parser.add_argument("-root", "--data_root", type = str, help = "the root of the dataset")

options = parser.parse_args()

import os
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn, optim
from tqdm import tqdm
from utils import *
from dataset import *


torch.manual_seed(0)
device=torch.device(f'cuda:{options.GPU_index}')

def train(
    model,
    trainset,
    testset,
    option,
    verbose=True,
):
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size = options.batch_size, 
        shuffle = True,
        num_workers = 4
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size = options.batch_size,
        shuffle = False,
        num_workers = 4
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr = options.learning_rate, 
        momentum = options.momentum, 
        weight_decay = options.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,], gamma=0.5)
    for epoch in tqdm(range(options.n_epoch)):
        model.train()
        Loss = []
        train_acc = 0
        test_acc = 0

        for idx, (image, label) in enumerate(trainloader):
            optimizer.zero_grad()
            image.requires_grad = True
            image = image.to(device)
            label = label.to(device)        

            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
            train_acc += (pred.max(1)[1] == label).float().sum().item()

        train_acc /= len(trainset)

        model.eval()
        with torch.no_grad():
            for idx, (image, label) in enumerate(testloader):
                image = image.to(device)
                label = label.to(device)
                pred = model(image)
                test_acc += (pred.max(1)[1] == label).float().sum().item()
            test_acc /= len(testset)

        print("Epoch: {}/{}".format(epoch + 1, options.n_epoch),
              "loss: {:.3f}".format(torch.mean(torch.FloatTensor(Loss)).item()),
              "train acc: {:.4f}".format(train_acc),
              "test acc: {:.4f}".format(test_acc),)
        scheduler.step()
        torch.save(model.state_dict(), f'model/{options.model_name}_CUB.pth')


if __name__ == '__main__':
    
    if not os.path.exists('model'):
        os.mkdir('model')
    if options.model_name == 'vgg':
        model = torchvision.models.vgg16_bn(pretrained = True).to(device)
        model.classifier[6] = nn.Linear(4096, 200).to(device)
    elif options.model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained = True).to(device)
        model.classifier[6] = nn.Linear(4096, 200).to(device)
    get_n_params(model)
    dataset = CUB(options.data_root, normalization=False)
    trainset = CUB(options.data_root, normalization=True, train_test='train')
    testset = CUB(options.data_root, normalization=True, train_test='test')
    train(model, trainset, testset, options, True)

    

    
