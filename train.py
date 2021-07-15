from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
#from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import torch.nn as nn
import copy
import models.starnet as starnet
import models.crnn as crnn
import string
from nltk.metrics import edit_distance
import torchvision
import cv2
import pdb
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--lan', required=True, help='language to train on')
parser.add_argument('--arch',required=True, help='select one of these - crnn or starnet')
parser.add_argument('--charlist',required=True, help='path to the character list')
parser.add_argument('--finetune', required=True, default=False, help='finetune on real data')
parser.add_argument('--savedir', default='tensorboard_runs', help='where to store the tensorboard logs')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet')
parser.add_argument('--expr_dir', default='output_results', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--deal_with_lossnan', action='store_true',help='whether to replace all nan/inf in gradients to zero')

opt = parser.parse_args()

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available() and opt.cuda:
    print('Nothing wrong with cuda')

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
test_dataset = dataset.lmdbDataset(root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))
assert train_dataset

if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    sampler=sampler,shuffle=True, num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

charlist_fname = opt.charlist 
opt.alphabet = open(charlist_fname,'r').readlines()
nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = nn.CTCLoss(zero_infinity=True)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.arch == 'crnn':
   crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
elif opt.arch == 'starnet':
   crnn = starnet.STARNET(opt.imgH, opt.imgW, nc, nclass, opt.nh)

crnn.apply(weights_init)
model_dict = crnn.state_dict()
if opt.pretrained != '':
    model_path = opt.pretrained
    checkpoint = torch.load(model_path)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    print('loading pretrained model from %s' % opt.pretrained)
    pretrained_dict = {k: v for k, v in checkpoint.items() if checkpoint[k].size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    crnn.load_state_dict(model_dict)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.LongTensor(opt.batchSize)

#opt.cuda = False
if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

if opt.deal_with_lossnan:
    if torch.__version__ >= '1.1.0':
        criterion = nn.CTCLoss(zero_infinity = True)
    else:
        crnn.register_backward_hook(crnn.backward_hook)

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    crnn.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    norm_ED = 0
    #max_iter = min(max_iter, len(data_loader))
    max_iter = len(data_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image, finetune=opt.finetune)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            target = target.lower().strip()
            gt = target.strip()
            if pred == target:
                n_correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    print("Samples Correctly recognised = " + str(n_correct))
    accuracy = n_correct / float(max_iter * opt.batchSize)
    crr = norm_ED / float(max_iter * opt.batchSize)
    lossval = loss_avg.val()
    print('Test loss: %f, accuracy: %f, crr: %f' % (loss_avg.val(), accuracy, crr))
    return lossval, accuracy, crr

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    optimizer.zero_grad()
    preds = crnn(image, finetune=opt.finetune)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    cost.backward()
    optimizer.step()
    return cost

losses_per_epoch = []
acc_per_epoch = []
best_acc = 0.0
is_best = 0
l_avg = utils.averager()

writer = SummaryWriter('{0}/runs_{1}_{2}'.format(opt.savedir,opt.lan,opt.arch))

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        l_avg.add(cost)
        i += 1
        writer.add_scalar('Loss/train', loss_avg.val(), epoch*len(train_loader) + i)

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            lossval, acc, crr = val(crnn, test_dataset, criterion)
            # pdb.set_trace()	
            writer.add_scalar('Loss/val', lossval, (epoch*len(train_loader)+i)/opt.valInterval)
            writer.add_scalar('Acc-WRR/accuracy_val', acc, (epoch*len(train_loader)+i)/opt.valInterval)
            writer.add_scalar('CRR/char_val', crr, (epoch*len(train_loader)+i)/opt.valInterval)

            is_best = acc >= best_acc
            if is_best:
                best_acc = acc
                filename = '{0}/best_model_{2}_{1}.pth'.format(opt.expr_dir, opt.arch, opt.lan)
                torch.save(crnn.state_dict(), filename)
                is_best = 0
