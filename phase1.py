# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.datasets import input_dataset
from models import *
import argparse, sys
import numpy as np
import datetime
import shutil
from random import sample 
from loss import loss_cross_entropy,loss_cores,f_beta

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.05)
parser.add_argument('--lr_plan', type = str, help = 'base, cyclic', default = 'cyclic')
parser.add_argument('--loss', type = str, help = 'ce, cores', default = 'cores')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='pairflip')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--ideal', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--model', type = str, help = 'cnn,resnet', default = 'cnn')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

def get_noise_pred(loss_div, args, epoch=-1, alpha=0.):
    #Get noise prediction
    print('DEBUG, loss_div', loss_div.shape)
    llast = loss_div[:, epoch]
    idx_last = np.where(llast>alpha)[0]
    print('last idx:', idx_last.shape)
    return idx_last    

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]/(1+f_beta(epoch))
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, num_classes, train_loader,model, optimizer,loss_all,loss_div_all,loss_type, noise_prior = None):
    train_total=0
    train_correct=0 
    print(f'current beta is {f_beta(epoch)}')
    v_list = np.zeros(num_training_samples)
    idx_each_class_noisy = [[] for i in range(num_classes)]
    if not isinstance(noise_prior, torch.Tensor):
        noise_prior = torch.tensor(noise_prior.astype('float32')).cuda().unsqueeze(0)
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        class_list = range(num_classes)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits = model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec
        if loss_type=='ce':
            loss = loss_cross_entropy(epoch,logits, labels,class_list,ind, noise_or_not, loss_all, loss_div_all)
        elif loss_type=='cores':
            loss, loss_v = loss_cores(epoch,logits, labels,class_list,ind, noise_or_not, loss_all, loss_div_all, noise_prior = noise_prior)
            v_list[ind] = loss_v
            for i in range(batch_size):
                if loss_v[i] == 0:
                    idx_each_class_noisy[labels[i]].append(ind[i])
        else:
            print('loss type not supported')
            raise SystemExit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(num_classes)]
    noise_prior_delta = np.array(class_size_noisy)
    print(noise_prior_delta)

    train_acc=float(train_correct)/float(train_total)
    return train_acc, noise_prior_delta

# Evaluate the Model
def evaluate(test_loader,model,save=False,epoch=0,best_acc_=0,args=None):
    model.eval()    # Change model to 'eval' mode.
    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total) 

    if save:
        if acc > best_acc_:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,args.loss + args.noise_type + str(args.noise_rate)+'best.pth.tar'))
            #np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_div_all_best.npy',loss_div_all)
            #np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_all_best.npy',loss_all)
            best_acc_ = acc
        if epoch == args.n_epoch -1:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,args.loss + args.noise_type + str(args.noise_rate)+'last.pth.tar'))
            #np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_div_all_last.npy',loss_div_all)
            #np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_all_best.npy',loss_all)
    return acc, best_acc_



#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 64
learning_rate = args.lr 

# load dataset
train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type,args.noise_rate)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model
print('building model...')
if args.model == 'cnn':
    model = CNN(input_channel=3, n_outputs=num_classes)
else:
    model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Creat loss and loss_div for each sample at each epoch
loss_all = np.zeros((num_training_samples,args.n_epoch))
loss_div_all = np.zeros((num_training_samples,args.n_epoch))
### save result and model checkpoint #######   
save_dir = args.result_dir +'/' +args.dataset + '/' + args.model 
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = batch_size, 
                                   num_workers=args.num_workers,
                                   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64, 
                                  num_workers=args.num_workers,
                                  shuffle=False)
alpha_plan = [0.1] * 50 + [0.01] * 50 
#alpha_plan = []
#for ii in range(args.n_epoch):
#    alpha_plan.append(learning_rate*pow(0.95,ii))
model.cuda()
txtfile=save_dir + '/' +  args.loss + args.noise_type + str(args.noise_rate) + '.txt' 
if os.path.exists(txtfile):
    os.system('rm %s' % txtfile)
with open(txtfile, "a") as myfile:
    myfile.write('epoch: train_acc test_acc \n')

epoch=0
train_acc = 0
best_acc_ = 0.0
#print(best_acc_)
# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
# train models
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc, noise_prior_delta = train(epoch,num_classes,train_loader, model, optimizer,loss_all,loss_div_all,args.loss,noise_prior = noise_prior_cur)
    noise_prior_cur = noise_prior*num_training_samples - noise_prior_delta
    noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)
# evaluate models
    test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model,epoch=epoch,best_acc_=best_acc_,args=args)
# save results
    #det_by_loss  = det_acc(save_dir,best_ratio,args,loss_all,noise_or_not,epoch,sum_epoch = False)
    #det_by_loss_div = det_acc(save_dir,best_ratio,args,loss_div_all,noise_or_not,epoch,sum_epoch = False)
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
    #print('precision of labels by loss is', det_by_loss)
    #print('precision of labels by loss div is', det_by_loss_div)
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' ' + str(test_acc) + "\n")
    np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_all.npy',loss_all)
    np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'loss_div_all.npy',loss_div_all)
    np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'noise_or_not.npy',noise_or_not)
    np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'train_noisy_labels.npy',train_dataset.train_noisy_labels)
    if epoch ==40:
        idx_last = get_noise_pred(loss_div_all, args, epoch=epoch)
        np.save(save_dir + '/' + args.loss + args.noise_type + str(args.noise_rate)+'_noise_pred.npy',idx_last)
