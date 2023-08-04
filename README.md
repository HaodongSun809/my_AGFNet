# my_AGFNet
the code
# my_AGFNet
the code
1. Requirements
   Pytorch 1.11; Python 3.8; Cuda 10.0;
2. Data Preparation
3. Download the datasets from 'https://pan.baidu.com/s/19EPBeyJLE1uZqyXBoVkK9Q'[code:8888] and trained model (output\SCNet_multi_V7\Net_epoch_best.pth) from here:https://pan.baidu.com/s/1IZlqBoHXUgLzD_3yFrGVIg [code:8888]. Then put them under the following directory:
   .idea
   dataset\ 
     -RGBD_test\
     -RGBD_train\
   model
   output
   pretrain_models
   test_result
     -SCNet_multi_V7
   utilsss
   eval.py
   evalold.py
   test.py
   train.py

4.Training & Testing
(1)Train the BBSNet:
    batchsize 4 --gpu_id 0 
(2)Test the BBSNet:
    The test maps will be saved to './test_result/SCNet_multi_V7/'. 
(3)Evaluate the result maps:
    You can evaluate the result maps using the 'eval.py'.
5.results:
   Test maps(test_result\SCNet_multi_V7) of our model can be download from 'https://pan.baidu.com/s/1WHsqKenIKBKuZYa2gyHTmg'[code:8888].
