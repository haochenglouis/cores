import logging
import os
import copy
import numpy as np
import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Subset, Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10
from augmentations import *
from common import get_logger
from samplers.stratified_sampler import StratifiedSampler
from utils import noisify

logger = get_logger('Unsupervised Data Augmentation')
logger.setLevel(logging.INFO)

class MyCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
        


def get_dataloaders(dataset, batch, batch_unsup, dataroot, with_noise=True, random_state=0, unsup_idx=set()):
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_valid = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    autoaug = transforms.Compose([])
    if isinstance(C.get()['aug'], list):
        logger.debug('augmentation provided.')
        autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
    else:
        logger.debug('augmentation: %s' % C.get()['aug'])
        if C.get()['aug'] == 'fa_reduced_cifar10':
            autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif C.get()['aug'] == 'autoaug_cifar10':
            autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif C.get()['aug'] == 'autoaug_extend':
            autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
        elif C.get()['aug'] == 'default':
            pass
        else:
            raise ValueError('not found augmentations. %s' % C.get()['aug'])
    transform_train.transforms.insert(0, autoaug)

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset in ['cifar10', 'cifar100']:
        if dataset == 'cifar10':
            total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
            unsup_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
            testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
            #nb_classes = 10
        elif dataset == 'cifar100':
            total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
            unsup_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=None)
            testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
            #nb_classes = 100
        else:
            raise ValueError
        print('DEBUG:', len(total_trainset.targets))
 
        if not with_noise:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
            sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
            train_idx, valid_idx = next(sss)
            #train_labels = [total_trainset.targets[idx] for idx in train_idx]   

            trainset = Subset(total_trainset, train_idx)        # for supervised
            #trainset.targets = train_labels  #THIS DOES NOTHING

            otherset = Subset(unsup_trainset, valid_idx)        # for unsupervised
            # otherset = unsup_trainset
            otherset = UnsupervisedDataset(otherset, transform_valid, autoaug, cutout=C.get()['cutout'])
        else:
            #import ipdb
            #ipdb.set_trace()

            #print('Noisy data config: noise_rate={}, noise_type={}'.format(noise_rate, noise_type))
            print('unsup_idx:', len(unsup_idx))

            all_idx = list(range(len(total_trainset)))
            sup_idx = [idx for idx in all_idx if idx not in unsup_idx]
            print('sup_idx: ', len(sup_idx))
            
            #apply noise to supervised trainset
            train_labels_with_noise = np.load(C.get()['train_labels']).reshape(-1)
            noisy_trainset = copy.deepcopy(total_trainset)
            noisy_trainset.targets = train_labels_with_noise

            train_labels_clean = np.array(total_trainset.targets)
            sup_labels_clean = np.array([train_labels_clean[idx] for idx in sup_idx]) #for estimating actual noise rate

            est_noise = len(np.where(train_labels_clean!=train_labels_with_noise)[0])
            print('noise labels total: ', est_noise)
            print('estimated noise rate: ', est_noise / len(train_labels_clean))              

            trainset = Subset(noisy_trainset, sup_idx)
            otherset = Subset(unsup_trainset, unsup_idx)

            sup_labels_with_noise = [trainset.dataset.targets[idx] for idx in sup_idx]
            est_noise = len(np.where(sup_labels_clean != np.array(sup_labels_with_noise))[0])
            print('noise labels in sup. data: ', est_noise)
            print('sup. data noise rate: ', est_noise / len(sup_labels_clean))

            trainset.targets = sup_labels_with_noise  #only for sampler
            otherset = UnsupervisedDataset(otherset, transform_valid, autoaug, cutout=C.get()['cutout'])
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=False,
        sampler=StratifiedSampler(trainset.targets), drop_last=True)

    unsuploader = torch.utils.data.DataLoader(
        otherset, batch_size=batch_unsup, shuffle=True, num_workers=8, pin_memory=False,
        sampler=None, drop_last=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=False,
        drop_last=False
    )
    return trainloader, unsuploader, testloader


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, transform_default, transform_aug, cutout=0):
        self.dataset = dataset
        self.transform_default = transform_default
        self.transform_aug = transform_aug
        self.transform_cutout = CutoutDefault(cutout)   # issue 4 : https://github.com/ildoonet/unsupervised-data-augmentation/issues/4

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        img1 = self.transform_default(img)
        img2 = self.transform_default(self.transform_aug(img))
        img2 = self.transform_cutout(img2)

        return img1, img2

    def __len__(self):
        return len(self.dataset)
