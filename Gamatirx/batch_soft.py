from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, models

from tqdm import tqdm
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import random
import numpy as np
import warnings
import torch.utils.data as data
from PIL import Image

plt.ion()  # interactive mode
warnings.filterwarnings('ignore')

class PlainDataset(Dataset):

    def __init__(self, split, labelled_root_dir, preTransform=None, postTransform=None):
        self.labelled_root_dir = labelled_root_dir
        # Output of pretransform should be PIL images
        self.preTransform = preTransform
        self.postTransform = postTransform
        self.split = split
        self.labelled_data_dir = labelled_root_dir + '/' + split
        self.dataset = datasets.ImageFolder(self.labelled_data_dir, self.preTransform) 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        plain_img, plain_class = self.dataset[index]
        if self.postTransform:
            sample = self.postTransform(plain_img)
        else:
            sample = transforms.ToTensor(plain_img)
        return sample, plain_class

# General Code for supervised train
def plaintrain(model, fc_layer, dataloaders, criterion, optimizer, scheduler, 
    device, checkpoint_path, file, saveinterval=1, last_epochs=0, num_epochs=20):
    
    since = time.time()
    best_acc = 0.0

    data_path = checkpoint_path+'/../Tensorboard'
    data_writer = SummaryWriter(logdir=data_path)

    for epoch in range(last_epochs, last_epochs+num_epochs):
        if epoch<10:
            criterion.flag = 0
        else:
            criterion.flag = 1
        print('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('\nEpoch {}/{} \n'.format(epoch, last_epochs+num_epochs - 1))
        file.write('-' * 10)
        file.write('\n')
        file.flush()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                fc_layer.train()
                dirname = 'data/TrainLoss_Plain'
            else:
                model.eval()  # Set model to evaluate mode
                fc_layer.eval()
                dirname = 'data/TestLoss_Plain'

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            end = time.time()

            # Iterate over data.
            for _, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = labels.size(0)
                n_samples += batchSize

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = fc_layer(model(inputs))
                    loss = criterion(outputs, labels)
                    true_loss = nn.CrossEntropyLoss()(outputs.detach(), labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                # running_loss += loss.item() * labels.size(0)
                running_loss += true_loss.item() * labels.size(0)
                pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()

            # Metrics
            top_1_acc = running_corrects / n_samples
            epoch_loss = running_loss / n_samples

            data_writer.add_scalars(dirname, {
                    'Acc': top_1_acc,
                    'EpochLoss': epoch_loss,
                }, epoch)

            print('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))

            file.write('{} Loss: {:.6f} Top 1 Acc: {:.6f} \n'.format(phase, epoch_loss, top_1_acc))
            file.flush()

            # deep copy the model
            if phase == 'test' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_fc_wts = copy.deepcopy(fc_layer.state_dict())
        if (epoch+1) % saveinterval == 0:
            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (checkpoint_path, epoch))
            torch.save(fc_layer.state_dict(), '%s/fc_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f} \n'.format(best_acc))
    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    file.write('Best test Acc: {:4f} \n'.format(best_acc))
    file.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    fc_layer.load_state_dict(best_fc_wts)
    return model, fc_layer


def plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers):
    image_datasets = {
        x: PlainDataset(x, data_root, data_pre_transforms[x], data_post_transforms[x])
        for x in ['train', 'test']
    }
    assert image_datasets
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            pin_memory=True, shuffle=True, num_workers=num_workers
        ) for x in ['train', 'test']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders



# Main
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default='', help="cuda : ?")

parser.add_argument('--batchsize', type=int, default=256, help="set batch size")
parser.add_argument('--numworkers', type=int, default=4, help="set num workers")
parser.add_argument('--lr_net', type=float, default=1e-2, help='learning rate, default=0.001')
parser.add_argument('--weight_net', type=float, default=1e-6, help="weight decay")
parser.add_argument('--lr_fc', type=float, default=1e-2, help='learning rate, default=0.001')
parser.add_argument('--weight_fc', type=float, default=1e-6, help="weight decay")

parser.add_argument('--netCont', default='', help="path to net (for continue training)")
parser.add_argument('--plainCont', default='', help="path to plain fc_layer (for continue training)")
parser.add_argument('--manualSeed', default=2077, type=int, help='manual seed')

parser.add_argument('--pretrain', type=int, default=1, help="pretrain on")

# opt = parser.parse_args(args=[])
opt = parser.parse_args()
# opt.netCont = './models/net_epoch_56.pth'

out_dir = '../Plain_{}/models'.format(opt.batchsize)
log_out_dir = '../Plain_{}'.format(opt.batchsize)

try:
    os.makedirs(out_dir)
except OSError:
    pass

file = open("{}/logs.txt".format(log_out_dir), "w+")
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000) 
file.write("Random Seed: {} \n".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


cudnn.benchmark = True
image_size = (224, 224)
# data_root = '~/Datasets/miniImageNet'
data_root = '~/Storage/Kaggle265'
batch_size = opt.batchsize      # 512, 256
num_workers = opt.numworkers    # 4

patch_dim = 96
contra_dim = 128
gap = 6
jitter = 6

saveinterval = 10
num_epochs = 200

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file.write("using " + str(device) + "\n")
file.flush()

# Initiate dataset and dataset transform
data_pre_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
    ]),
}
data_post_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
}

loader_plain = plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
# Model Initialization
# 仅支持 Res-Net !!!
model_all = models.resnet18(pretrained=opt.pretrain)
num_ftrs = model_all.fc.in_features

def make_MLP(input_ftrs, hidden_ftrs, output_ftrs, layers=1):
    modules_list = []
    modules_list.append(nn.Flatten())
    if layers==1:
        modules_list.append(nn.Linear(input_ftrs, output_ftrs))
    else:
        modules_list.append(nn.Linear(input_ftrs, hidden_ftrs))
        for _ in range(layers-2):
            modules_list.append(nn.LeakyReLU())
            modules_list.append(nn.Linear(hidden_ftrs, hidden_ftrs))
        modules_list.append(nn.LeakyReLU())
        modules_list.append(nn.Linear(hidden_ftrs, output_ftrs))
    
    modules_list.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*modules_list)


model_ft = nn.Sequential(*(list(model_all.children())[:-1]))
fc_plain = make_MLP(num_ftrs, num_ftrs, output_ftrs=265, layers=1)

if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)
    fc_plain = nn.DataParallel(fc_plain)

model_ft = model_ft.to(device)
fc_plain = fc_plain.to(device)


# Load state : model & fc_layer
def loadstate(model, fc_layer, net_Cont, fc_Cont, device, file):
    if net_Cont != '':
        model.load_state_dict(torch.load(net_Cont, map_location=device))
        print('Loaded model state ...')
        file.write('Loaded model state ...')

    if fc_Cont != '':
        fc_layer.load_state_dict(torch.load(fc_Cont, map_location=device))
        print('Loaded fc_layer state ...')
        file.write('Loaded fc_layer state ...')

loadstate(model_ft, fc_plain, opt.netCont, opt.plainCont, device, file)


# Model trainer
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.flag = 1
    def forward(self, x, y):
        if self.flag == 1:
            vector = nn.CrossEntropyLoss(reduction='none')(x,y)
            power = torch.softmax(vector.detach(), 0)
            return torch.sum(vector * power)
        else:
            return nn.CrossEntropyLoss()(x,y)
        # vector = nn.CrossEntropyLoss(reduction='none')(x,y)
        # power = torch.softmax(vector, 0)
        # Ret = vector * power
        # return torch.sum(Ret) + torch.mean(vector)

# criterion = nn.CrossEntropyLoss()
criterion = My_loss()

milestones = [10, 20, 40, 80, 160, 200]
milegamma = 0.6

optimizer = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_plain.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)


model_ft, fc_plain = plaintrain(
    model_ft, fc_plain, 
    loader_plain, criterion, optimizer, scheduler, 
    device, out_dir, file, saveinterval, 0, num_epochs
)

