import pandas as pd
import cv2


bbox_info=[1449, 538, 1479 ,605]
temp_path='/home/mayank_sati/Desktop/one_Shot_learning/image_xs/1566366024136726396_369144.535422_4321493.08002.jpg'
img = cv2.imread(temp_path)
frame = img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[3])]
frame = img[(bbox_info[1]):(bbox_info[3]), (bbox_info[0]):(bbox_info[3])]


xmin=int(bbox_info[0])
xmax=int(bbox_info[2])
ymin=int(bbox_info[1])
ymax=int(bbox_info[3])
# crop_img = img[y:y+h, x:x+w]
crop_img = img[538:605, 1449:1479]
crop_img1 = img[ymin:ymax, xmin:xmax]
print(1)