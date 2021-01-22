import numpy as np
noise='symmetric'
nr=0.2
alpha=-4

save_dir = '../results/cifar10/resnet'
loss=np.load('../results/cifar10/resnet/uspl{}{}loss_div_all.npy'.format(noise,nr))
results='../results/cifar10/resnet/uspl{}{}.txt'.format(noise,nr)

lines = open(results, 'r').readlines()[1:]
best = 0.
best_epoch = -1
for line in lines:
    epoch, _, testacc=line.strip().split()
    testacc = float(testacc)
    if testacc > best:
        best=testacc
        best_epoch = int(epoch[:-1])
best_epoch = 21
print('best_epoch ', best_epoch, best)

lbest = loss[:,best_epoch]
llast = loss[:,-1]

idx_best = np.where(lbest>alpha)[0]
print('best idx:', idx_best.shape)

idx_last = np.where(llast>alpha)[0]
print('last idx:', idx_last.shape)

np.save(save_dir+'/usplpeer{}{}best_noise_pred.npy'.format(noise, nr), idx_best)
np.save(save_dir+'/usplpeer{}{}last_noise_pred.npy'.format(noise, nr), idx_last)


    

