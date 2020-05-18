from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *
from qatm_pytorch_custom import CreateModel, ImageDataset,nms_multi,nms,run_multi_sample,plot_result_multi,run_one_sample
import pandas as pd
import natsort
# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys
import time
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

    #######################################333333
    # root = "/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0003"
    template_root="/home/mayank_sati/Desktop/one_Shot_learning/xshui"
    
    ref_id=1003

    # input_folder='/home/mayank_sati/Desktop/one_Shot_learning/image_xs'
    input_folder='/home/mayank_sati/Desktop/one_Shot_learning/fresh_crop'
    # input_folder='/home/mayank_sati/Desktop/one_Shot_learning/xshui'
    for root, _, filenames in os.walk(input_folder):
        filenames = natsort.natsorted(filenames, reverse=False)

        if (len(filenames) == 0):
            print("Input folder is empty")
        # time_start = time.time()
        for filename in filenames:
            t1 = time.time()
            # print(filename)
            title, ext = os.path.splitext(os.path.basename(filename))
            time_stamp = title.split("_")[0]
            x_pos = title.split("_")[1]
            y_pos = title.split("_")[2]
            temp_name,bbox_info=find_template(ref_id,[float(x_pos),float(y_pos)])

            ####################################################################
            # saving template
            temp_path =template_root+"/"+temp_name
            # temp_path =root+"/"+filename
            img = cv2.imread(temp_path)
            frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
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
            scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            # scores, w_array, h_array, thresh_list=run_one_sample(model, dataset['template'], dataset['image'], dataset['image_name'])
            print("nms...")
            print(" time taken in running model", (time.time() - t1) * 1000)
            boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
            # time_take=(time.time()-t)*1000
            print("time taken nms", (time.time() - t1) * 1000)
            # print('amount of time taken',time_take)
            _ = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            print("result.png was saved")