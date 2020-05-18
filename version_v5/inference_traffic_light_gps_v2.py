#this  implementation mainly work on grey scale image
import argparse
import shutil
from pathlib import Path

import natsort
from qatm_pytorch_v3 import CreateModel, ImageDataset,ImageDataset_2, plot_result_mayank, nms, run_one_sample_mayank
from torchvision import models

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
    # combo1
    #  ref_id=1001....1005

    template_root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train"
    input_folder = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test'
    # input_folder = '/media/mayank_sati/DATA/datasets/gwm_specific/xshui_all/e-w-n-2_crop'
    # input_folder = '/media/mayank_sati/DATA/datasets/gwm_specific/xshui_all/e-w-n-2 (copy)'
    csv_name = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/farm_all.csv'

    ref_id = 1004
    #######################################333333

    #######################################333333
    all_centre=[]

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
            # frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
            # frame = img[int(bbox_info[1]-35):int(bbox_info[3]+35), int(bbox_info[0]-35):int(bbox_info[2]+35)]
            # frame = img[int(bbox_info[1]-20):int(bbox_info[3]+25), int(bbox_info[0]-25):int(bbox_info[2]+25)]
            temp_scale_2 =40
            # frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
            # frame = img[int(bbox_info[1]-50):int(bbox_info[3]+100), int(bbox_info[0]-50):int(bbox_info[2]+50)]
            frame = img[int(bbox_info[1] - temp_scale_2):int(bbox_info[3] + temp_scale_2),
                             int(bbox_info[0] - temp_scale_2):int(bbox_info[2] + temp_scale_2)]
            ##########################3
            # cv2.putText(frame, "templ", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.imshow('img_frame', frame)
            ch = cv2.waitKey(10)
            if ch & 0XFF == ord('q'):
                cv2.destroyAllWindows()
            # cv2.waitKey(1)
            cv2.destroyAllWindows()



            # frame = img[int(bbox_info[1]-15):int(bbox_info[3]+15), int(bbox_info[0]-15):int(bbox_info[2]+15)]
            # frame = img[int(y[value][1]):int(y[value][3]), int(y[value][0]):int(y[value][2])]
            # frame = img[int(y[value][1]-40):int(y[value][3]+40), int(y[value][0]-30):int(y[value][2]+30)]
            cv2.imwrite("../data/cust_template/myimage.jpg", frame)
            # print(" time taken in template processing", (time.time() - t1) * 1000)
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
            dataset = ImageDataset_2(Path(template_dir), image_path, thresh_csv='../thresh_template.csv')
            # print(" time taken in data processing", (time.time() - t1) * 1000)

            # print("calculate score...")

            # print(" time taken in loading model", (time.time() - t1) * 1000)
            # scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            scores, w_array, h_array, thresh_list = run_one_sample_mayank(model, dataset)
            # scores, w_array, h_array, thresh_list=run_one_sample(model, dataset['template'], dataset['image'], dataset['image_name'])
            # print("nms...")
            # print(" time taken in running model", (time.time() - t1) * 1000)
            # boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
            # # time_take=(time.time()-t)*1000
            # # print('amount of time taken',time_take)
            # _ = plot_result_multi(model,dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            thresh_list = .99
            boxes,_,max_score = nms(scores, w_array, h_array, thresh_list)
            limit_boxes = 5
            boxes = boxes[:limit_boxes]
            # _ = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result.png')
            # _ = plot_result_mayank(dataset.image_raw, boxes, show=True, save_name='result.png')
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
            for all_cen in all_centre:
                for center in all_cen:
                    point = (int(center[0]), int(center[1]))
                    cv2.circle(img=imgt, center=point, radius=2, color=(0, 200, 0), thickness=2)

            # cv2.imshow('img_frame', imgt)
            # ch = cv2.waitKey(5)
            # if ch & 0XFF == ord('q'):
            #     cv2.destroyAllWindows()
            # # cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # cv2.imwrite("../data/all_point.jpg", imgt)
            # cv2.imwrite("../data/all/all_point_"+str(time.time())+".jpg", imgt)
            # 

            output_folder='/home/mayank_sati/Documents/temp_result_'
            output_folder=output_folder+str(ref_id)
            if not os.path.exists(output_folder):
                print("Output folder not present. Creating New folder...")
                os.makedirs(output_folder)
            print(counter)
            cv2.imwrite(output_folder+"/all_point_"+str(time.time())+".jpg", imgt)