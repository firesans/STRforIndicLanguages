import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
from nltk.metrics.distance import edit_distance
import argparse
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import sys
from PIL import Image
import numpy as np
import io
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model to test on')
parser.add_argument('--test_data', required=True, help='path to testing dataset')
parser.add_argument('--lexicon', required=True, help='path to the lexicon file')
parser.add_argument('--type', required=True, help='Choose CRNN or STAR-Net')
opt = parser.parse_args()
print(opt)

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=2,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= self.nSamples, 'index range error'

        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(str(img_key).encode())
            #print(type(imgbuf))  
            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf)
                img = img.convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model_path = opt.model
lexicon_filename = opt.lexicon
p = open(lexicon_filename, 'r').readlines()
alphabet = p

nclass = len(p) + 1
print(nclass)

converter = utils.strLabelConverter(alphabet)

if opt.type == 'crnn':
    import models.crnn as crnn
    model = crnn.CRNN(32, 1, nclass, 256)
elif opt.type == 'star':
    import models.crnn_new as crnn
    model = starnet.STARNET(32, 100, 1, nclass, 256)

if torch.cuda.is_available():
    model = model.cuda()

model.apply(weights_init)
model_dict = model.state_dict()
checkpoint = torch.load(model_path)
for key in list(checkpoint.keys()):
  if 'module.' in key:
    checkpoint[key.replace('module.', '')] = checkpoint[key]
    del checkpoint[key]

model_dict.update(checkpoint)
model.load_state_dict(checkpoint)

vocab = []
for i in alphabet:
  vocab.append(i.strip())

def testeval(img):
 
    transformer = dataset.resizeNormalize((100, 32))
    image = transformer(img)
    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.squeeze(1)
  
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return img, sim_pred

def test_batch(lmdbData):
    
    nwrr = 0
    nSamples = lmdbData.__len__()
    norm_ED = 0
    nwrr = 0
    print(nSamples)
    non = 0
    curr = 0
    non_chars = {}
    for i in range(nSamples):
        img, lab = lmdbData.__getitem__(i)
        flag = 0
        gt = lab
        curr += 1
        im, pred = testeval(img)
        pred = pred.strip()
        gt = gt.strip()

        if(pred.strip() == gt.strip()):
            nwrr += 1
        if len(gt) == 0 or len(pred) == 0:
            norm_ED += 0
        elif len(gt) > len(pred):
            norm_ED += 1 - edit_distance(pred, gt) / len(gt)
        else:
            norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    
    print(nwrr)
    wrr = nwrr / float(nSamples) * 100
    crr = norm_ED / float(nSamples) * 100

    return wrr, crr

lmdb_data = opt.test_data
lm = lmdbDataset(lmdb_data) 
wrr, crr = test_batch(lm)
print("FINAL RESULTS")
print("Word Recognition Rate :" + str(wrr))
print("Character Recognition Rate :" + str(crr))
