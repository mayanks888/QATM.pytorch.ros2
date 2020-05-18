from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *
from qatm_pytorch_custom_color import CreateModel, ImageDataset,plot_result_mayank,nms_multi,nms,run_one_sample_mayank,run_multi_sample,plot_result_multi,run_one_sample
import pandas as pd
import natsort
import copy
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys
import time
from PIL import Image
from data_preprocess_for_inference import find_template
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda',default=True, action='store_true')
    parser.add_argument('-s', '--sample_image', default='data/sample/gwm_1284.jpg')
    # parser.add_argument('-t', '--template_images_dir', default='/home/mayank_sati/Desktop/qatm_data/template/')
    parser.add_argument('-t', '--template_images_dir', default='data/cust_template/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()

    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    counter=0
    #######################################333333
    # combo1
    #  ref_id=1001....1005

    template_root = "/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/e-w-n-1"
    input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/e-w-n-2'
    # input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/e-w-n-2_crop'
    # input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/e-w-n-2 (copy)'
    csv_name = 'e-w-n-1.csv'

    ref_id = 1001
    #######################################333333

    #######################################333333
    # combo2
    #  ref_id=1012....1021

    # template_root = "/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-1"
    # input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-2_crop'
    # csv_name = 's-n-1.csv'
    # ref_id = 1013


    #######################################333333
   # combo3
  #  ref_id=1012....1021

    template_root="/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-1"
    input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-2_crop'
    input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-3'
    # input_folder = '/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/S-N-2-CROP_2'
    # input_folder = '/home/mayank_sati/Desktop/one_Shot_learning/RANDOM_XS'
    csv_name='s-n-1.csv'
    ref_id=1012
    #######################################333333


    # input_folder='/home/mayank_sati/Desktop/one_Shot_learning/image_xs'
    # input_folder='/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/e-w-n-l-2'

    # input_folder='/home/mayank_sati/Desktop/one_Shot_learning/xshui'
    for root, _, filenames in os.walk(input_folder):
        filenames = natsort.natsorted(filenames, reverse=False)

        if (len(filenames) == 0):
            print("Input folder is empty")
        # time_start = time.time()
        for filename in filenames:
            counter+=1
            if not counter % 4==0:
                continue
            t1 = time.time()
            print(filename)
            title, ext = os.path.splitext(os.path.basename(filename))
            time_stamp = title.split("_")[0]
            x_pos = title.split("_")[1]
            y_pos = title.split("_")[2]
            temp_name,bbox_info=find_template(ref_id,[float(x_pos),float(y_pos)],csv_name)
            if temp_name==1:
                print("waiting for image with id ",ref_id)

                imgk=cv2.imread(os.path.join(root,filename))
                cv2.putText(imgk, "IDNOT FOUND", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                cv2.imshow("img5", imgk)
                # cv2.imshow('img', img)
                ch = cv2.waitKey(150)
                if ch & 0XFF == ord('q'):
                    cv2.destroyAllWindows()
                # cv2.waitKey(1)
                cv2.destroyAllWindows()
                continue
            ####################################################################
            # saving template
            temp_path =template_root+"/"+temp_name
            # temp_path =root+"/"+filename
            
            img = cv2.imread(temp_path)
            frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
            # frame = img[int(bbox_info[1]-0):int(bbox_info[3]+100), int(bbox_info[0]-35):int(bbox_info[2]+35)]
            # frame = img[int(bbox_info[1]-20):int(bbox_info[3]+25), int(bbox_info[0]-25):int(bbox_info[2]+25)]

            ##########################3
            # cv2.putText(frame, "templ", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.imshow('img_frame', frame)
            ch = cv2.waitKey(500)
            if ch & 0XFF == ord('q'):
                cv2.destroyAllWindows()
            # cv2.waitKey(1)
            cv2.destroyAllWindows()



            # frame = img[int(bbox_info[1]-15):int(bbox_info[3]+15), int(bbox_info[0]-15):int(bbox_info[2]+15)]
            # frame = img[int(y[value][1]):int(y[value][3]), int(y[value][0]):int(y[value][2])]
            # frame = img[int(y[value][1]-40):int(y[value][3]+40), int(y[value][0]-30):int(y[value][2]+30)]
            cv2.imwrite("data/cust_template/myimage.jpg", frame)
            print(" time taken in template processing", (time.time() - t1) * 1000)
    ################################################
            template_dir = args.template_images_dir
            # image_path = args.sample_image
            # path=''
            # image_path = path
            # image_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0001/1559324322534832565.jpeg'
            # image_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0001/1559324320268203434.jpeg'
            image_path =root+"/"+filename
            # image_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0003/1559324341001511289.jpeg'
            # image_path = 'result.jpeg'
            dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
            print(" time taken in data processing", (time.time() - t1) * 1000)

            print("calculate score...")

            # print(" time taken in loading model", (time.time() - t1) * 1000)
            # scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            scores, w_array, h_array, thresh_list = run_one_sample_mayank(model, dataset)
            # scores, w_array, h_array, thresh_list=run_one_sample(model, dataset['template'], dataset['image'], dataset['image_name'])
            print("nms...")
            print(" time taken in running model", (time.time() - t1) * 1000)
            # boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
            # # time_take=(time.time()-t)*1000
            # # print('amount of time taken',time_take)
            # _ = plot_result_multi(model,dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            thresh_list = .2
            boxes = nms(scores, w_array, h_array, thresh_list)
            # _ = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            _ = plot_result_mayank(dataset.image_raw, boxes, show=True, save_name='result.png')
            print("result.png was saved")
