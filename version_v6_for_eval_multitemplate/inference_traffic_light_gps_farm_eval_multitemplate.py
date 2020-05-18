#this  implementation mainly work on grey scale image
import argparse
import shutil
from pathlib import Path
import pandas as pd
import natsort
from qatm_pytorch_v3 import CreateModel, ImageDataset,ImageDataset_2, plot_result_mayank, nms, run_one_sample_mayank,\
    run_multi_sample,nms_multi,plot_result_multi,plot_result_multi_univ
from torchvision import models
from data_preprocess_for_inference import find_template,find_template_2,find_template_with_temp_info
from utils import *
from collections import namedtuple
import os
import pandas as pd

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
    temp_save_path="../data/cust_template"
    template_root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train"
    input_folder = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/evaluation/valid_farm'
    csv_name = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train.csv'
    # ref_id = 1002
    show=False
    df = pd.read_csv(csv_name)
    all_centre = []
    temp_info={}
    #######################################333333
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    csv_path = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test_scaled_eval_farm.csv'
    data = pd.read_csv(csv_path)
    print(data.head())


    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    grouped = split(data, 'filename')

    for group in sorted(grouped):
            filename = group.filename
            obj_ids = list(group.object["obj_id"])
            counter+=1
            if not counter % 10==0:
                continue
            print("current count is ",counter)
            t1 = time.time()
            print("current sample image is ",filename)
            title, ext = os.path.splitext(os.path.basename(filename))
            time_stamp = title.split("_")[0]
            x_pos = title.split("_")[1]
            y_pos = title.split("_")[2]

            #creating all the template from the csv file we have recieved tagged to the images
            for index, temp_ref_id in enumerate(obj_ids):
                temp_name,bbox_info,_=find_template_2(temp_ref_id,[float(x_pos),float(y_pos)],df)
                print("current template image is ", temp_name)
                if temp_name==1:
                    print("waiting for image with id ",temp_ref_id)
                    if show:
                        imgk=cv2.imread(os.path.join(input_folder,filename))
                        cv2.putText(imgk, "ID NOT FOUND", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                        cv2.imshow("img5", imgk)
                        # cv2.imshow('img', img)
                        ch = cv2.waitKey(0)
                        if ch & 0XFF == ord('q'):
                            cv2.destroyAllWindows()
                        cv2.destroyAllWindows()
                    continue
                # temp_info[temp_name]=bbox_info
                current_temp = str(temp_ref_id)  + ".jpg"
                dict={current_temp:list(bbox_info)}
                temp_info.update(dict)
                ####################################################################
                # saving template
                temp_path =template_root+"/"+temp_name
                img = cv2.imread(temp_path)
                temp_scale_2 =100
                # frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
                # frame = img[int(bbox_info[1]-50):int(bbox_info[3]+100), int(bbox_info[0]-50):int(bbox_info[2]+50)]
                frame = img[int(bbox_info[1] - temp_scale_2):int(bbox_info[3] + temp_scale_2),
                                 int(bbox_info[0] - temp_scale_2):int(bbox_info[2] + temp_scale_2)]
                ##########################3
                if show:
                    cv2.imshow('img_frame', frame)
                    ch = cv2.waitKey(1000)
                    if ch & 0XFF == ord('q'):
                        cv2.destroyAllWindows()
                    # cv2.waitKey(1)
                    cv2.destroyAllWindows()
                # current_temp="mytemp_"+str(index)+".jpg"
                current_temp_path=temp_save_path+"/"+current_temp
                cv2.imwrite(current_temp_path, frame)


    ################################################
            template_dir = args.template_images_dir
            image_path =input_folder+"/"+filename
            dataset = ImageDataset_2(Path(template_dir), image_path, thresh_csv='../thresh_template.csv')

            scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            ############################################################################33
            boxes, indices,scores = nms_multi(scores, w_array, h_array, thresh_list)
            if len(boxes.shape) == 3:
                # limit_boxes = 5
                # boxes = boxes[:limit_boxes]
                d_img = plot_result_multi_univ(dataset.image_raw, boxes, indices, show=True, save_name='result_sample.png')
            ################################################################################
            # scores, w_array, h_array, thresh_list = run_one_sample_mayank(model, dataset)
            # scores, w_array, h_array, thresh_list=run_one_sample(model, dataset['template'], dataset['image'], dataset['image_name'])
            # print("nms...")
            # # time_take=(time.time()-t)*1000
            # # print('amount of time taken',time_take)
            # _ = plot_result_multi(model,dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            # thresh_list = .5
            # boxes,_,max_score = nms(scores, w_array, h_array, thresh_list)
            # limit_boxes = 5
            # boxes = boxes[:limit_boxes]
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
###########################################################################33333
            # ploting bounding box form template size for all the selected template
            for index_1,temp_no in enumerate(indices):
                temp_1=str(dataset.template_path[temp_no])
                temp_=temp_1.split("/")[-1]
                bbox_val=temp_info[temp_]
                mycenter=centr_box[index_1]
                # temp_h=bbox_val[3]-bbox_val[1]
                # temp_w=bbox_val[2]-bbox_val[0]

                temp_h = (bbox_val[3] - bbox_val[1]) / 2
                temp_w = (bbox_val[2] - bbox_val[0]) / 2
                xmin=int (mycenter[0]-(temp_w/2))
                xmax=int (mycenter[0]+(temp_w/2))
                ymin=int (mycenter[1]-(temp_h/2))
                ymax=int (mycenter[1]+(temp_h/2))
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # top = (y[read_index][0], y[read_index][3])
                # bottom = (y[read_index][2], y[read_index][1])
                image_scale = dataset.image_raw
                top=(xmin,ymax)
                bottom=(xmax,ymin)
                cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
                # cv2.putText(image_scale, y[read_index][4], ((y[read_index][0]+y[read_index][2])/2, y[read_index][1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
                # cv2.putText(image_scale, str(temp_), ((xmin + xmax) / 2, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), lineType=cv2.LINE_AA)

            if 1:
                cv2.imshow('img_frame', image_scale)
                ch = cv2.waitKey(1000)
                if ch & 0XFF == ord('q'):
                    cv2.destroyAllWindows()
                # cv2.waitKey(1)
                cv2.destroyAllWindows()
                # cv2.imwrite("../data/all/all_point_" + str(time.time()) + ".jpg", image_scale)


'''
          all_centre.append(centr_box)
          ############################3
            #this was done to check the centre of the bounding box
            imgt=dataset.image_raw
            # cv2.circle(img=frame, center=point, radius=radius, color=circle_color, thickness=line_width)
            for all_cen in all_centre:
                for center in all_cen:
                    point = (int(center[0]), int(center[1]))
                    cv2.circle(img=imgt, center=point, radius=2, color=(0, 200, 0), thickness=2)
            if show:
                cv2.imshow('img_frame', imgt)
                ch = cv2.waitKey(5)
                if ch & 0XFF == ord('q'):
                    cv2.destroyAllWindows()
                # cv2.waitKey(1)
                cv2.destroyAllWindows()
                cv2.imwrite("../data/all/all_point_"+str(time.time())+".jpg", imgt)


            output_folder='/home/mayank_sati/Documents/farm_eval'
            # output_folder=output_folder+str(ref_id)
            if not os.path.exists(output_folder):
                print("Output folder not present. Creating New folder...")
                os.makedirs(output_folder)
            cv2.imwrite(output_folder+"/all_point_"+str(time.time())+".jpg", imgt)'''