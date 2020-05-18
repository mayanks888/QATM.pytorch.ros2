"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

from __future__ import print_function
from PIL import Image

import os
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import time
import argparse

import torch
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from numpy import linalg as LA
from torch.autograd import Variable
# from utils import TinyImageNetLoader

from net import *



def find_correct_template_output(trainloader, testloader, is_gpu):
    """
    Calculate accuracy for TripletNet model.

        1. Form 2d array: Number of training images * size of embedding
        2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
        3. Perform subtraction between the two 2D arrays
        4, Take L2 norm of the 2d array (after subtraction)
        5. Get the 30 min values (argmin might do the trick)
        6. Repeat for the rest of the embeddings in the test set

    """
    net = TripletNet(resnet101())

    # For training on GPU, we need to transfer net and data onto the GPU
    # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
    # if is_gpu:
    #     net = torch.nn.DataParallel(net).cuda()
    #     cudnn.benchmark = True

    print('==> Retrieve model parameters ...')
    # checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    checkpoint = torch.load("/home/mayank_sati/Desktop/git/2/AI/QATM/version_v3/checkpoint/checkpoint.pth.tar")
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(torch.load("../checkpoint/checkpoint.pt"))

    net.eval()
    net.cuda()

    t1 = time.time()
    # dictionary of test images with class
    # class_dict = get_classes()
    t2 = time.time()
    print("Get all test image classes, Done ... | Time elapsed {}s".format(t2 - t1))

    # list of traning images names, e.g., "../tiny-imagenet-200/train/n01629819/images/n01629819_238.JPEG"
    training_images = []
    # for line in open("../triplets.txt"):
    #     line_array = line.split(",")
    #     training_images.append(line_array[0])
    t3 = time.time()
    # print("Get all training images, Done ... | Time elapsed {}s".format(t3 - t2))

    # get embedded features of training
    embedded_features = []
    with torch.no_grad():
        for batch_idx, (data1, data2, data3) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output
            embedded_a, _, _ = net(data1, data2, data3)
            embedded_a_numpy = embedded_a.data.cpu().numpy()

            embedded_features.append(embedded_a_numpy)

    t4 = time.time()
    print("Get embedded_features, Done ... | Time elapsed {}s".format(t4 - t3))

    # TODO: 1. Form 2d array: Number of training images * size of embedding
    embedded_features_train = np.concatenate(embedded_features, axis=0)

    embedded_features_train.astype('float32').tofile(
        "../embedded_features_train.txt")

    # embedded_features_train = np.fromfile("../embedded_features_train.txt", dtype=np.float32)

    # TODO: 2. For a single test embedding, repeat the embedding so that it's the same size as the array in 1)
    with torch.no_grad():
        for test_id, test_data in enumerate(testloader):

            if test_id % 5 == 0:
                print("Now processing {}th test image".format(test_id))

            if is_gpu:
                test_data = test_data.cuda()
            test_data = Variable(test_data)

            embedded_test, _, _ = net(test_data, test_data, test_data)
            embedded_test_numpy = embedded_test.data.cpu().numpy()

            embedded_features_test = np.tile(embedded_test_numpy, (embedded_features_train.shape[0], 1))

            # TODO: 3. Perform subtraction between the two 2D arrays
            embedding_diff = embedded_features_train - embedded_features_test

            # TODO: 4, Take L2 norm of the 2d array (after subtraction)
            # embedding_norm = LA.norm(embedding_diff, axis=0)
            embedding_norm = LA.norm(embedding_diff, axis=1)
            print(embedding_norm)
            # TODO: 5. Get the 30 min values (argmin might do the trick)
            # min_index = embedding_norm.argsort()[:30]
            min_index = np.argmax(embedding_norm)
            return(min_index)
            # TODO: 6. Repeat for the rest of the embeddings in the test set
            accuracies = []

            # get test image class
            # test_image_name = "val_" + str(test_id) + ".JPEG"
            # test_image_class = class_dict[test_image_name]
            #
            # # for each image results in min distance
            # for i, idx in enumerate(min_index):
            #     if test_id % 5 == 0 and i % 5 == 0:
            #         print("\tNow processing {}th result of test image".format(i))
            #
            #     correct = 0
            #
            #     # get result image class
            #     top_result_image_name = training_images[idx]
            #     top_result_image_name_class = top_result_image_name.split(
            #         "/")[3]
            #
            #     if test_image_class == top_result_image_name_class:
            #         correct += 1
            #
            # acc = correct / 30
            # accuracies.append(acc)
    #
    # t5 = time.time()
    # print("Get embedded_features, Done ... | Time elapsed {}s".format(t5 - t4))
    #
    # with open('accuracies.txt', 'w') as f:
    #     for acc in accuracies:
    #         f.write("%s\n" % acc)
    #
    # print(sum(accuracies))
    # print(len(accuracies))
    # avg_acc = sum(accuracies) / len(accuracies)
    # print("Test accuracy {}%".format(avg_acc))


def image_loader(path):
    """Image Loader helper function."""
    return Image.open(path.rstrip("\n")).convert('RGB')


class TripletImageLoader(Dataset):
    """Image Loader for Tiny ImageNet."""

    def __init__(self, base_path, train=True, loader=image_loader):
        """
        Image Loader Builder.

        Args:
            base_path: path to triplets.txt
            filenames_filename: text file with each line containing the path to an image e.g., `images/class1/sample.JPEG`
            triplets_filename: A text file with each line containing three images
            transform: torchvision.transforms
            loader: loader for each image
        """
        self.transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.transform_train = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # Normalize test set same as training set without augmentation
        self.transform_test = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.base_path = base_path
        # self.transform_train = transform
        self.loader = loader

        self.train_flag = train
        self.image_name_sample=[]
        self.image_name_query=[]

        # load training data
        if self.train_flag:
            triplets = []
            # test_images = os.listdir(os.path.join("../mayank_datasets", "train", "Images"))
            test_images = os.listdir('/home/mayank_sati/Desktop/git/2/AI/QATM/data/image_save')
            for test_image in test_images:
                loaded_image = self.loader(os.path.join("/home/mayank_sati/Desktop/git/2/AI/QATM/data/image_save", test_image))
                triplets.append(loaded_image)

                self.image_name_sample.append(test_image)
            self.triplets = triplets
            # return self.image_name_sample
            # triplets = []
            # for line in open(triplets_filename):
            #     line_array = line.split(",")
            #     triplets.append((line_array[0], line_array[1], line_array[2]))
            # self.triplets = triplets

        # load test data
        else:
            singletons = []
            test_images = os.listdir('/home/mayank_sati/Desktop/git/2/AI/QATM/data/cust_template_2')
            for test_image in test_images:
                loaded_image = self.loader(os.path.join("/home/mayank_sati/Desktop/git/2/AI/QATM/data/cust_template_2", test_image))
                singletons.append(loaded_image)
                self.image_name_query.append(test_image)
            self.singletons = singletons
            # return self.image_name_query

    def __getitem__(self, index):
        """Get triplets in dataset."""
        # get trainig triplets
        if self.train_flag:

            img = self.triplets[index]
            if self.transform_train is not None:
                img = self.transform_train(img)
            return img,img,img

        # get test image
        else:
            img = self.singletons[index]
            if self.transform_test is not None:
                img = self.transform_test(img)
            return img

    def __len__(self):
        """Get the length of dataset."""
        if self.train_flag:
            return len(self.triplets)
        else:
            return len(self.singletons)

    def get_image_list(self):
        return self.image_name_sample


def TinyImageNetLoader(root, batch_size_train, batch_size_test):
    """
    Tiny ImageNet Loader.

    Args:
        train_root:
        test_root:
        batch_size_train:
        batch_size_test:

    Return:
        trainloader:
        testloader:
    """
    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Loading Tiny ImageNet dataset
    print("==> Preparing Tiny ImageNet dataset ...")

    trainset = TripletImageLoader(base_path=root, triplets_filename="../triplets.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, num_workers=32)

    testset = TripletImageLoader(base_path=root, triplets_filename="", transform=transform_test, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, num_workers=32)

    return trainloader, testloader