#this  implementation mainly work on grey scale image
import argparse
import shutil
from pathlib import Path
import pandas as pd
import natsort
from qatm_pytorch_v3 import CreateModel, ImageDataset,ImageDataset_2, plot_result_mayank, nms, run_one_sample_mayank
from torchvision import models
from data_preprocess_for_inference import find_template,find_template_2,increase_bounding_box_scale_diff_apr
from utils import *

# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import time
from data_preprocess_for_inference import find_template
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda',default=True, action='store_true')
    parser.add_argument('-s', '--sample_image', default='data/sample/gwm_1284.jpg')
    # parser.add_argument('-t', '--template_images_dir', default='/home/mayank_sati/Desktop/qatm_data/template/')
    parser.add_argument('-t', '--template_images_dir', default='../data/cust_template/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()

    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    counter=0
    #######################################333333
    template_root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train"
    input_folder = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test'
    input_folder = '/home/mayank_sati/Desktop/k'
    csv_name = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/farm_all.csv'
    ref_id = 1019

    show=True
    df = pd.read_csv(csv_name)
    #######################################333333

    all_centre=[]
    for root, _, filenames in os.walk(input_folder):
        filenames = natsort.natsorted(filenames, reverse=False)

        if (len(filenames) == 0):
            print("Input folder is empty")
        # time_start = time.time()
        for filename in filenames:
            counter+=1
            if not counter % 4==0:
                continue
            print("current count is ",counter)
            t1 = time.time()
            print("current sample image is ",filename)
            title, ext = os.path.splitext(os.path.basename(filename))
            time_stamp = title.split("_")[0]
            x_pos = title.split("_")[1]
            y_pos = title.split("_")[2]
            temp_name,bbox_info,min_dis_val=find_template_2(ref_id,[float(x_pos),float(y_pos)],df)
            print("current template image is ", temp_name)
            if temp_name==1:
                print("waiting for image with id ",ref_id)
                if show:
                    imgk=cv2.imread(os.path.join(root,filename))
                    cv2.putText(imgk, "ID NOT FOUND", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                    cv2.imshow("img5", imgk)
                    # cv2.imshow('img', img)
                    ch = cv2.waitKey(5)
                    if ch & 0XFF == ord('q'):
                        cv2.destroyAllWindows()
                    cv2.destroyAllWindows()
                continue
            ####################################################################
            # saving template
            temp_path =template_root+"/"+temp_name
            img = cv2.imread(temp_path)
            # temp_scale_2 =80


            # frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
            # frame = img[int(bbox_info[1]-50):int(bbox_info[3]+100), int(bbox_info[0]-50):int(bbox_info[2]+50)]
            # frame = img[int(bbox_info[1] - temp_scale_2):int(bbox_info[3] + temp_scale_2),
            #                  int(bbox_info[0] - temp_scale_2):int(bbox_info[2] + temp_scale_2)]

            ################################33
            # custom scaling
            # increase_bounding_box_scale_diff_apr
            scale_width=6
            scale_height=4

            if min_dis_val < 1:
                scale_widh = .1
                scale_height = .1

            bbox_info=increase_bounding_box_scale_diff_apr(img,bbox_info,scale_width,scale_height)
            frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
            #########################3
            if show:
                cv2.imshow('img_frame', frame)
                ch = cv2.waitKey(1000)
                if ch & 0XFF == ord('q'):
                    cv2.destroyAllWindows()
                # cv2.waitKey(1)
                cv2.destroyAllWindows()
            cv2.imwrite("../data/cust_template/myimage.jpg", frame)
    ################################################
            template_dir = args.template_images_dir
            image_path =root+"/"+filename
            dataset = ImageDataset_2(Path(template_dir), image_path, thresh_csv='../thresh_template.csv')

            # scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            scores, w_array, h_array, thresh_list = run_one_sample_mayank(model, dataset)
            # scores, w_array, h_array, thresh_list=run_one_sample(model, dataset['template'], dataset['image'], dataset['image_name'])
            # print("nms...")
            # # time_take=(time.time()-t)*1000
            # # print('amount of time taken',time_take)
            # _ = plot_result_multi(model,dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            thresh_list = .99
            boxes,_,max_score = nms(scores, w_array, h_array, thresh_list)
            limit_boxes = 5
            boxes = boxes[:limit_boxes]
            if show:
                # _ = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result.png')
                _ = plot_result_mayank(dataset.image_raw, boxes, show=True, save_name='result.png')
                # print("result.png was saved")

            #################################################3333
            # here I will start with the next version of template matching algorithm
            # finding the center of the boxes
            x_y_min=boxes[:,0]#this is the array of all x min and y min in boxes
            x_y_max=boxes[:,1]#this is a array of  al x max and y max in boxes
            diff=x_y_max-x_y_min#finding xmax - xmin and ymax - ymin
            centr_box=x_y_min+diff/2
            all_centre.append(centr_box)
          ############################3
            #this was done to check the centre of the bounding box
            imgt=dataset.image_raw
            # cv2.circle(img=frame, center=point, radius=radius, color=circle_color, thickness=line_width)
            for index,all_cen in enumerate(all_centre[-7:]):
                for center in all_cen:
                    point = (int(center[0]), int(center[1]))
                    cv2.circle(img=imgt, center=point, radius=2, color=(0, 200, 0), thickness=2)
                # if index>=5:
                #     break


            if show:
                cv2.imshow('img_frame', imgt)
                ch = cv2.waitKey(5000)
                if ch & 0XFF == ord('q'):
                    cv2.destroyAllWindows()
                # cv2.waitKey(1)
                cv2.destroyAllWindows()
                cv2.imwrite("../data/all/all_point_"+str(time.time())+".jpg", imgt)


            output_folder='/home/mayank_sati/Documents/farm_last_6/'
            output_folder=output_folder+str(ref_id)
            if not os.path.exists(output_folder):
                print("Output folder not present. Creating New folder...")
                os.makedirs(output_folder)
            cv2.imwrite(output_folder+"/all_point_"+str(time.time())+".jpg", imgt)