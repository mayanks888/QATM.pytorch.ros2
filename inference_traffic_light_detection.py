from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *
from qatm_pytorch_custom import CreateModel, ImageDataset,nms_multi,nms,run_multi_sample,plot_result_multi
import pandas as pd
# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image', default='data/sample/gwm_1284.jpg')
    # parser.add_argument('-t', '--template_images_dir', default='/home/mayank_sati/Desktop/qatm_data/template/')
    parser.add_argument('-t', '--template_images_dir', default='data/cust_template/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    
    
    querry_image_id=1004
    flag=False

    #######################################333333
    root = "/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0003"
    # csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
    # csv_path = '/home/mayank_sati/Desktop/git/2/AI/Annotation_tool_V3/system/Labels/traffic_shot_2019-05-31-13-38-28_4_0002.csv'
    csv_path = '/home/mayank_sati/Desktop/git/2/AI/Annotation_tool_V3/system/Labels/traffic_shot_2019-05-31-13-38-28_4_0003.csv'
    # saving_path="/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/pycharm_work/annotated/"
    data = pd.read_csv(csv_path)
    mydata = data.groupby('img_name')
    # print(data.groupby('class').count())
    len_group = mydata.ngroups
    mygroup = mydata.groups
    # new = data.groupby(['img_name'])['class'].count()
    ###############################################3333
    x = data.iloc[:, 0].values
    y = data.iloc[:, 5:10].values
    for da in mygroup.keys():
        Flag=False
        index = mydata.groups[da].values
        for value in index:
            if y[value][-1]==querry_image_id:
                print(querry_image_id)
                tmp=y[value][:-1]
                path=root+"/"+da
                img=cv2.imread(path)
                # frame = img[int(y1):int(y2), int(x1):int(x2)]
                frame = img[int(y[value][1]):int(y[value][3]), int(y[value][0]):int(y[value][2])]
                # frame = img[int(y[value][1]-40):int(y[value][3]+40), int(y[value][0]-30):int(y[value][2]+30)]
                cv2.imwrite("data/cust_template/myimage.jpg", frame)
                flag=True
                break
        if flag==True:
            break


    ################################################
    template_dir = args.template_images_dir
    # image_path = args.sample_image
    image_path = path
    # image_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0001/1559324322534832565.jpeg'
    # image_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0001/1559324320268203434.jpeg'
    image_path = '/home/mayank_sati/Desktop/Screenshot from 2019-08-30 11-04-17.jpeg'
    image_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_2019-05-31-13-38-28_4_0003/1559324341001511289.jpeg'
    image_path = 'result.jpeg'
    dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
    t=time.time()
    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    print("calculate score...")
    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
    print("nms...")
    boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
    # time_take=(time.time()-t)*1000
    print("actual time taken", (time.time() - t) * 1000)
    # print('amount of time taken',time_take)
    _ = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result.png')
    print("result.png was saved")


