# STRforIndicLanguages
PyTorch implementation of [Scene Text Recognition for Transfer Learning in Indic Languages](https://cvit.iiit.ac.in/images/ConferencePapers/2021/Transfer_Learning_Indian_STR.pdf). 
We received Best Paper Award for this work at CBDAR'21.

## Dependence

- Python3.6.5
- torch==1.2.0
- torchvision==0.4.0
- tensorboard==2.3.0

## Train your data

### Prepare data

- Follow the instructions in [meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>) to create lmdb datasets
	Use the same step to create separate training and validation synthetic data.

### Change parameters and alphabets

Please update the parameters and alphabets according to your requirement. 

- Change parameters in the train.py file accordingly
- Input the list of characters (or alphabets) to the train.py file wrt to the language you are training on : 
	The charlist input to the train.py should contain all the characters corresponding to the language present in the dataset, else the program will throw a key error during the training process.

### Train

Run `mytrain.py` --

```sh
python3 mytrain.py --trainRoot <\path to the train lmdb dataset> \
--valRoot <\path to the validation lmdb dataset> \
--arch <\architecture:CRNN/STARNET> --lan <\language> --charlist <\path to the character text file> \
--batchSize 32 --nepoch 15 --cuda --expr_dir <\path to the output experiments directory> \
--displayInterval 10 --valInterval 100 --adadelta \ 
--manualSeed 1234 --random_sample --deal_with_lossnan 
```

## Reference

[meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>) \
[Sierkinhane/crnn_chinese_characters_rec](<https://github.com/Sierkinhane/crnn_chinese_characters_rec>)
