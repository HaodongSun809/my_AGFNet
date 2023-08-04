import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy import misc
import cv2
from utilsss.data_val import test_dataset
from model import BBSNet,SCNet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size——352——416——512')
parser.add_argument('--pth_path', type=str, default='./output/structure_loss/SCNet_multi_V7/0.02508.pth')
parser.add_argument("-nocrf", "--nocrf", action="store_false")
opt = parser.parse_args()


####for _data_name in['LFSD', 'NJU2K', 'NLPR','RGBD135', 'SIP', 'STERE']:
for _data_name in ['LFSD', 'NJU2K', 'NLPR', 'STERE']:
    data_path = './dataset/RGBD_test/{}'.format(_data_name)
    save_path = './test_result_0.02508/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
#    save_path_coarse = './test_result_mine/{}/{}/coarse/'.format(opt.pth_path.split('/')[-2], _data_name)##############################
    model = SCNet.SCNet_multi_V7().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(save_path_coarse, exist_ok=True)
    image_root = '{}/RGB/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    depth_root = '{}/depth/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, depth, name, img_for_post = test_loader.load_data()
        img_for_post = np.array(img_for_post)
        # print(img_for_post.shape)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        output = model(image,depth)[1]

        res = F.upsample(output, size=[img_for_post.shape[0],img_for_post.shape[1]], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res * 255

#        coarse = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)##########################
 #       coarse = coarse.sigmoid().data.cpu().numpy().squeeze()##########################
    #    coarse = (coarse - coarse.min()) / (coarse.max() - coarse.min() + 1e-8)###########################
  #      coarse = coarse * 255############################
        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        # cv2.imwrite(save_path + name,res*255)
        cv2.imwrite(save_path + name, res)
  #      cv2.imwrite(save_path_coarse + name, coarse)####################################
