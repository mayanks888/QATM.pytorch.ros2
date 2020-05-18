import pandas as pd
import numpy as np
import cv2
# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
# df = pd.read_csv("xshui (copy).csv")
# # data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
#
# 
# ref_id=1001
# ref_pos=[369152.294564,	4321492.85272]
#
# grouped = df.groupby('img_name',sort=True)
# mygroup = df.groupby(['obj_id'],sort=True)
# # return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
# mydata = mygroup.groups
# concerned_list=mydata[ref_id].values
# data=df.values[concerned_list]
# ###############################33333
# # calculating eucldean distance
# x_dif=abs(data[:,10]- ref_pos[0])
# y_dif=abs(data[:,11]- ref_pos[1])
# total_dif=x_dif+y_dif
# #######################################
# index=np.argmin(total_dif)
# img_name=data[index][0]
# bbox=data[index][5:9]
# print(img_name)
# print(bbox)
#


def find_template(ref_id,ref_pos,csv_name):
    # df = pd.read_csv("xshui (copy).csv")
    # csv_path="/media/mayank_sati/DATA/datasets/gwm_specific/xshui_all/"+csv_name
    csv_path=csv_name
    df = pd.read_csv(csv_path)

    # df = pd.read_csv("/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-1.csv")
    # data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')

    # ref_id = 1001
    # ref_pos = [369152.294564, 4321492.85272]

    # grouped = df.groupby('img_name', sort=True)
    mygroup = df.groupby(['obj_id'], sort=True)
    # return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    mydata = mygroup.groups
    concerned_list = mydata[ref_id].values
    data = df.values[concerned_list]
    ###############################33333
    # calculating eucldean distance
    x_dif = abs(data[:, 10] - ref_pos[0])
    y_dif = abs(data[:, 11] - ref_pos[1])
    total_dif = x_dif + y_dif

    min_dist_val=np.amin(total_dif)
    print("min distane", min_dist_val)
    if min_dist_val > 10:
        # print(np.amin(total_dif)
        # flag=True
        return 1,1
    #######################################
    index = np.argmin(total_dif)
    img_name = data[index][0]
    bbox = data[index][5:9]
    # print(img_name)
    # print(bbox)q
    return(img_name,bbox)

def find_template_2(ref_id,ref_pos, newdf=""):
    # df = pd.read_csv("xshui (copy).csv")
    # csv_path="/media/mayank_sati/DATA/datasets/gwm_specific/xshui_all/"+csv_name
    # csv_path=csv_name
    # df = pd.read_csv(csv_path)
    df=newdf
    # df = pd.read_csv("/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-1.csv")
    # data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')

    # ref_id = 1001
    # ref_pos = [369152.294564, 4321492.85272]

    # grouped = df.groupby('img_name', sort=True)
    mygroup = df.groupby(['obj_id'], sort=True)
    # return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    mydata = mygroup.groups
    concerned_list = mydata[ref_id].values
    data = df.values[concerned_list]
    ###############################33333
    # calculating eucldean distance
    x_dif = abs(data[:, 10] - ref_pos[0])
    y_dif = abs(data[:, 11] - ref_pos[1])
    total_dif = x_dif + y_dif

    min_dist_val=np.amin(total_dif)
    index = np.argmin(total_dif)
    min_x_dist =x_dif[index]
    min_y_dist =y_dif[index]
    img_name = data[index][0]
    bbox = data[index][5:9]

    print("nearest template is ",img_name,)
    print("min x ",min_x_dist)
    print("min y  ",min_y_dist)
    print("min distane", min_dist_val)


    # if min_dist_val > 5:
    #     # print(np.amin(total_dif)
    #     # flag=True
    #     return 1,1

    # if ((min_x_dist <2 or min_y_dist <2) and min_dist_val < 10):

    # if min_x_dist > 5 and  min_y_dist > 5:
        # print(np.amin(total_dif)
        # flag=True
    return (img_name, bbox, min_dist_val)
    # else:
    #   return 1, 1,1
    #######################################
    # index = np.argmin(total_dif)
    # img_name = data[index][0]
    # bbox = data[index][5:9]
    # print(img_name)
    # print(bbox)q
    # return(img_name,bbox)



    #######################################
    # index = np.argmin(total_dif)
    # img_name = data[index][0]
    # bbox = data[index][5:9]
    # print(img_name)
    # print(bbox)q
    # return(img_name,bbox)

def increase_bounding_box_scale(img,mybbox,scale_width,scale_height):
    print("old box",mybbox)
    bbox_info=mybbox
    height, width, depth = img.shape

    # x_y_min = bbox_info[:, 0]  # this is the array of all x min and y min in boxes
    # x_y_max = bbox_info[:, 1]  # this is a array of  al x max and y max in boxes
    # diff = x_y_max - x_y_min  # finding xmax - xmin and ymax - ymin

    centr_box = (int((bbox_info[0] + bbox_info[2])/2),int((bbox_info[1] + bbox_info[3])/2))
    temp_width=   bbox_info[2] - bbox_info[0]
    temp_height= bbox_info[3] - bbox_info[1]

    bbox_info[1]=int(bbox_info[1]-(scale_height*(temp_height/2)))
    bbox_info[3]=int(bbox_info[3]+(scale_height*(temp_height/2)))

    bbox_info[0]=int(bbox_info[0]-(scale_width*(temp_width/2)))
    bbox_info[2]=int(bbox_info[2]+(scale_width*(temp_width/2)))

    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    if bbox_info[1] < 0:
        bbox_info[3]=bbox_info[3]+abs(bbox_info[1])
        bbox_info[1] = 0

    if bbox_info[3]>height:
        bbox_info[1]=bbox_info[1]-abs(bbox_info[1]-height)
        bbox_info[3]=height

    if bbox_info[0] < 0:
        bbox_info[2] = bbox_info[2] + abs(bbox_info[0])
        bbox_info[0] = 0

    if bbox_info[2] > width:
        bbox_info[0] = bbox_info[0] - abs(bbox_info[2] - width)
        bbox_info[3] = width

    return bbox_info

def  increase_bounding_box_scale_diff_apr(img,mybbox,scale_width,scale_height):
    # print("old box",mybbox)
    bbox_info=mybbox
    height, width, depth = img.shape

    # x_y_min = bbox_info[:, 0]  # this is the array of all x min and y min in boxes
    # x_y_max = bbox_info[:, 1]  # this is a array of  al x max and y max in boxes
    # diff = x_y_max - x_y_min  # finding xmax - xmin and ymax - ymin
    centr_box = (int((bbox_info[0] + bbox_info[2])/2),int((bbox_info[1] + bbox_info[3])/2))
    temp_width=   bbox_info[2] - bbox_info[0]
    temp_height= bbox_info[3] - bbox_info[1]

    bbox_info[1]=int(bbox_info[1]-(scale_height*(temp_height/2)))
    bbox_info[3]=int(bbox_info[3]+(scale_height*(temp_height/2)))

    bbox_info[0]=int(bbox_info[0]-(scale_width*(temp_width/2)))
    bbox_info[2]=int(bbox_info[2]+(scale_width*(temp_width/2)))

    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    if bbox_info[1] < 0:
        bbox_info[3]=bbox_info[3]-abs(bbox_info[1])
        bbox_info[1] = 0

    if bbox_info[3]>height:
        bbox_info[1]=bbox_info[1]+abs(bbox_info[1]-height)
        bbox_info[3]=height

    if bbox_info[0] < 0:
        bbox_info[2] = bbox_info[2] - abs(bbox_info[0])
        bbox_info[0] = 0

    if bbox_info[2] > width:
        bbox_info[0] = bbox_info[0] + abs(bbox_info[2] - width)
        bbox_info[2] = width

    return bbox_info

#
if __name__ == '__main__':
    img=cv2.imread("version_v5/1569609627426492150_301402.801308_4704820.4038.jpg")
    img=cv2.imread("/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train/1569609650059600932_301402.125496_4704662.06752.jpg")
    bb_old=[1045,205,1208,392]

    # bb_old=[1067, 512, 1097, 545]
    # bb_old=[1067+700, 512, 1097+700, 545]

    scale_widh=5
    scale_height=1

    top = (bb_old[0], bb_old[3])
    bottom = (bb_old[2], bb_old[1])
    cv2.rectangle(img, pt1=top, pt2=bottom, color=(255, 0, 0), thickness=2)


    # box=inc_temp_ratio(img,bb_old,scale_widh,scale_height)
    box=increase_bounding_box_scale_diff_apr(img,bb_old,scale_widh,scale_height)
    # print("old box",bb_old)
    print('newbox=',box)


    top = (box[0], box[3])
    bottom = (box[2], box[1])
    cv2.rectangle(img ,pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)


    show=True

    if show:
        cv2.imshow('img_frame', img)
        ch = cv2.waitKey(10000)
        if ch & 0XFF == ord('q'):
            cv2.destroyAllWindows()
        # cv2.waitKey(1)
        cv2.destroyAllWindows()
    # cv2.imwrite("../data/cust_template/myimage.jpg", frame)