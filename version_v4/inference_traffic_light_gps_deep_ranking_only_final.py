#this  implementation mainly work on grey scale image
import argparse
import shutil
from pathlib import Path

import natsort
from qatm_pytorch_v3 import CreateModel, ImageDataset,ImageDataset_2, plot_result_mayank, nms, run_one_sample_mayank
from torchvision import models
import torch
from utils import *
from imageloader_mayank import TripletImageLoader,TinyImageNetLoader,find_correct_template_output
# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import time
from data_preprocess_for_inference import find_template

if __name__ == '__main__':

    batch_size_train=7
    batch_size_test=7
    #deep ranking starts here
    root=''
    # trainloader, testloader = TinyImageNetLoader(args.dataroot, args.batch_size_train, args.batch_size_test)
    trainset = TripletImageLoader(base_path=root)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, num_workers=4)

    image_list_sample=trainset.image_name_sample



    testset = TripletImageLoader(base_path=root, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, num_workers=4)


    image_index=find_correct_template_output(trainloader, testloader, "is_gpu")
    final_image=image_list_sample[image_index]
    print (image_list_sample)
    save_frame_path = "../data/image_save/" + final_image
    max_score_img=cv2.imread(save_frame_path)
    cv2.imshow('final selection', max_score_img)
    ch = cv2.waitKey(2000)
    if ch & 0XFF == ord('q'):
        cv2.destroyAllWindows()
    # cv2.waitKey(1)
    cv2.destroyAllWindows()





