python3 mytrain.py --trainRoot /ssd_scratch/cvit/sanjana/hindi-train-lmdb \
--valRoot /ssd_scratch/cvit/sanjana/hindi-test-lmdb \
--arch crnn --lan hindi --charlist /ssd_scratch/cvit/sanjana/crnn_new/lexicon.txt \
--batchSize 32 --nepoch 15 --cuda --expr_dir /ssd_scratch/cvit/sanjana \
--displayInterval 10 --valInterval 100 --adadelta \ 
--manualSeed 1234 --random_sample --deal_with_lossnan 
