from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *
from qatm_pytorch_custom import CreateModel, ImageDataset,nms_multi,nms,run_multi_sample,plot_result_multi

# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    # parser.add_argument('-s', '--sample_image', default='sample/sample1.jpg')
    parser.add_argument('-s', '--sample_image', default='sample/sample1.jpg')
    # parser.add_argument('-t', '--template_images_dir', default='/home/mayank_sati/Desktop/qatm_data/template/')
    parser.add_argument('-t', '--template_images_dir', default='template/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    
    template_dir = args.template_images_dir
    image_path = args.sample_image
    dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
    
    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    print("calculate score...")
    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
    print("nms...")
    boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
    _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
    print("result.png was saved")


