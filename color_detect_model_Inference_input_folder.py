from __future__ import print_function

import os
import cv2
import time
import PIL.ImageOps

# from __future__ import print_function
import argparse

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, utils
import torchvision
import random
from PIL import Image
import torch
import copy
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class featx(nn.Module):
    def __init__(self):
        super(featx, self).__init__()
        model = models.vgg19(pretrained=True).features
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:8]
        for param in self.model.parameters():
            param.requires_grad = False
        # if self.use_cuda:
        #     self.model = self.model.cuda()
        # self.model[2].register_forward_hook(self.save_feature1)
        # self.model[16].register_forward_hook(self.save_feature2)

        self.fc1 = nn.Linear(15 * 15 * 128, 1000)
        # self.fc1 = nn.Linear(800, 1000)
        self.fc2 = nn.Linear(1000, 64)
        self.fc3 = nn.Linear(64, 4)


    def forward(self, x):
        out = self.model(x)
        # out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc1(out))
        out = self.fc3(out)
        # return out
        return F.log_softmax(out, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(5 * 5 * 32, 1000)
        # self.fc1 = nn.Linear(800, 1000)
        self.fc2 = nn.Linear(1000, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc1(out))
        out = self.fc3(out)
        # return out
        return F.log_softmax(out, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#


class Create_Image_Datasets(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # img0_tuple = random.choice(self.imageFolderDataset.samples)

        img0 = Image.open(img0_tuple[0])
        img0 = img0.convert("RGB")
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            # img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
        # return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        label = img0_tuple[1]
        return img0, label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class Config():
    # training_dir = "/home/mayank_sati/Desktop/label_traffic_light/base_color/training/"
    # # training_dir = "/home/mayank_sati/Desktop/label_traffic_light/roi_label_bins/"
    # # testing_dir = "/home/mayank_sati/Desktop/label_traffic_light/base_color/testing/"
    # testing_dir = "/home/mayank_sati/Desktop/label_traffic_light/roi_label_bins/"
    # training_dir = "/home/mayank_sati/pycharm_projects/pytorch/siamese/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/only_traffic_light/training/"
    training_dir = "/home/mayank_sati/Desktop/traffic_light/sorting_light/final_sort"
    testing_dir = "/home/mayank_sati/Desktop/traffic_light/sorting_light/test_data"
    # train_batch_size = 64
    # train_number_epochs = 3
    # train_batch_size = 64
    # train_number_epochs = 2

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
########################################33
    # model = Net().to(device)
    # model.load_state_dict(torch.load("color_model.pt"))
    #######################################################
    model = featx().to(device)
    model.load_state_dict(torch.load("color_model_19_on_All_tl_gwm.pt"))

    model.eval()
    ########################################33
    #this is to play with model selection
    model_2 = featx().to(device)

    #############################3333
    beta = 0.5  # The interpolation parameter
    params1_old = model.named_parameters()
    params2_old = model_2.named_parameters()

    dict_params2 = dict(params2_old)
    dict_params1 = dict(params1_old)

    # for name1, param1 in params1:
    #     if name1 in dict_params2:
    #         dict_params2[name1].data.copy_(beta * param1.data + (1 - beta) * dict_params2[name1].data)
    #
    # model.load_state_dict(dict_params2)
    #

    for name1, param2 in params2_old:
        if name1 in dict_params1:
            dict_params2[name1].data.copy_(beta * param2.data + (1 - beta) * dict_params1[name1].data)

    model_2.load_state_dict(dict_params1)

    params1 = model.named_parameters()
    params2 = model_2.named_parameters()

    dict_params2 = dict(params2)
    dict_params1 = dict(params1)

    ###################################
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ##################################33
    test_folder_dataset = dset.ImageFolder(root=Config.testing_dir)
    # input_folder='/home/mayank_sati/Desktop/traffic_light/sorting_light/all_train_images'
    input_folder='/home/mayank_sati/Desktop/one_Shot_learning/col'
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            return 1
        # time_start = time.time()
        for filename in filenames:
            file_path = (os.path.join(root, filename));

            # ################################33
            img0 = Image.open(file_path)
            img0 = img0.convert("RGB")
            transform = transforms.Compose([transforms.Resize((30, 30)), transforms.ToTensor(),normalize])
            img = transform(img0)
            img=img.unsqueeze(0)
            # input_batch = img0.repeat(ref_batch, 1, 1, 1)
            ############################
            img = img.to(device)
            t1=time.time()
            # output = model(img)
            output = model_2(img)
            print("actual time taken", (time.time() - t1) * 1000)
            data = torch.argmax(output, dim=1)
            # print(output)
            light_color = test_folder_dataset.classes[int(data)]
            image=cv2.imread(file_path)
            cv2.putText(image, light_color, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            ############################################################################################

            cv2.imshow('img', image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()