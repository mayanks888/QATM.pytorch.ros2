import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from seaborn import color_palette
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
from utils import *
from PIL import Image
# %matplotlib inline
from color_function import color_Detect
# # CONVERT IMAGE TO TENSOR

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_path = list(template_dir_path.iterdir())
        self.image_name = image_name
        
        self.image_raw = cv2.imread(self.image_name)
        # self.image_raw = self.image_raw[self.image_raw.shape[0] / 2:self.image_raw.shape[0]]
        height, width, channels = self.image_raw.shape
        # self.image_raw = self.image_raw[int(height / 2):height, 0:width]  # this line crops
        # self.image_raw =self.image_raw[0:int(height / 2), 0:int(width / 2)]
        # self.image_raw =self.image_raw[0:int(height / 2), 0:int(width)]

        self.thresh_df = None
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)
            
        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)
        
    def __len__(self):
        return len(self.template_names)
    
    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template = cv2.imread(template_path)
        if self.transform:
            template = self.transform(template)
        # thresh = 0.7
        thresh = .99
        if self.thresh_df is not None:
            if self.thresh_df.path.isin([template_path]).sum() > 0:
                thresh = float(self.thresh_df[self.thresh_df.path==template_path].thresh)

        ######################################
        # cv2.imshow('datasets_frame', self.image_raw)
        # ch = cv2.waitKey(20000)
        # if ch & 0XFF == ord('q'):
        #     cv2.destroyAllWindows()
        # # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        #####################################
        return {'image': self.image, 
                    'image_raw': self.image_raw, 
                    'image_name': self.image_name,
                    'template': template.unsqueeze(0), 
                    'template_name': template_path, 
                    'template_h': template.size()[-2],
                   'template_w': template.size()[-1],
                   'thresh': thresh}


# template_dir = 'template/'
# image_path = 'sample/sample1.jpg'
# dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
#

# ### EXTRACT FEATURE

class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model= copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)

        ######################################################333
        self.col_model = self.model[:8]
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # if self.use_cuda:
        #     self.model = self.model.cuda()
        # self.model[2].register_forward_hook(self.save_feature1)
        # self.model[16].register_forward_hook(self.save_feature2)


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # # def _load_pretrained(model, url, inchans=3):
        # #     state_dict = model_zoo.load_url(url)
        # model_2 = featx()
        # # model_2.load_state_dict(torch.load("color_model_19.pt"))
        # model_2.load_state_dict(torch.load("color_model_19_on_All_tl_gwm.pt"))
        # state_dict = self.model
        # inchans = 1
        # if inchans == 1:
        #     conv1_weight = state_dict['conv1.weight']
        #     state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        # elif inchans != 3:
        #     assert False, "Invalid number of inchans for pretrained weights"
        # model.load_state_dict(state_dict)
        ###################################################################333

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

                ###################################3333


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

        model_2 = featx()
        # model_2.load_state_dict(torch.load("color_model_19.pt"))
        model_2.load_state_dict(torch.load("../color_model_19_on_All_tl_gwm.pt"))
        # model_2 = featx().to(device)
        weights_FC1=model_2.fc1.weight.data
        weights_FC2=model_2.fc2.weight.data
        weights_FC3=model_2.fc3.weight.data

        self.fc1 = nn.Linear(15 * 15 * 128, 1000)
        self.fc1.weight.data = weights_FC1.data
        # self.fc1.weight = weights
        # self.fc1 = nn.Linear(800, 1000)
        self.fc2 = nn.Linear(1000, 64)
        self.fc2.weight.data = weights_FC2.data
        self.fc3 = nn.Linear(64, 4)
        self.fc3.weight.data = weights_FC3.data

        self.col_seq=nn.Sequential(
            self.col_model,
            nn.Linear(15 * 15 * 128, 1000),
            nn.Linear(1000, 64),
            nn.Linear(64, 4))
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@222
        # model = featx().to(device)
        # model.load_state_dict(torch.load("color_model_19.pt"))
        # model_2 = featx().to(device)

        # beta = 0.5  # The interpolation parameter
        # cool = model.named_parameters()
        # params1 = model.named_parameters()
        # params2 = model_2.named_parameters()
        #
        # dict_params2 = dict(params2)
        # dict_params1 = dict(params1)
        #
        #
        # for name1, param2 in params2:
        #     if name1 in dict_params1:
        #         dict_params2[name1].data.copy_(beta * param2.data + (1 - beta) * dict_params1[name1].data)
        #
        # model_2.load_state_dict(dict_params1)
        #
        # params1 = model.named_parameters()
        # params2 = model_2.named_parameters()
        #
        # dict_params2 = dict(params2)
        # dict_params1 = dict(params1)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def run_color(self, x):
        out = self.col_model(x)
        # out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out=out.to("cpu")
        # out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc1(out))
        out = self.fc3(out)
        # return out
        return F.log_softmax(out, dim=1)

    def run_color_2(self, x):
        out = self.col_seq(x)
        return F.log_softmax(out, dim=1)
        #############################################################

    #this is just like the feature pyramid function used to get more spacial information
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        
    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
            #model run here(forward function implemented here)
        _ = self.model(input)
        if mode=='big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        else:        
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)





class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _ , H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]


class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None
    def __call__(self, template, image, image_name):
        T_feat = self.featex(template)
        if self.I_feat_name is not image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            #this is where normilisation was done on the image feature as well as template feature.
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            #this one is to unite the two matrix together in the given way
            dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True), T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps


class QATM():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row*ref_col, qry_row*qry_col)
        #substracting row with max value in ref column and same we will do with querry column
        #
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        # cosine similarity function
        confidence = torch.sqrt(F.softmax(self.alpha*xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        #top tk is will give the top value in each row with index
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row*ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()
        #this is to filter the confidence with highest value of querry values
        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values
    def compute_output_shape( self, input_shape ):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)


# # NMS AND PLOT

# ## SINGLE

def nms(score, w_ini, h_ini, thresh=0.7):
    score=score.squeeze()
    dots = np.array(np.where(score > thresh*score.max()))
    
    x1 = dots[1] - w_ini//2
    x2 = x1 + w_ini
    y1 = dots[0] - h_ini//2
    y2 = y1 + h_ini

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = score[dots[0], dots[1]]
    order = scores.argsort()[::-1]

    if order.size > 0:
        max_score = scores[order[0]]
    else:
        max_score=0
    keep = []
    while order.size > 0:

        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.3)[0]
        order = order[inds + 1]
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes,max_score


def plot_result(image_raw, boxes, show=False, save_name=None, color=(255, 0, 0)):
    # plot result
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 3)
    if show:
        plt.imshow(d_img)
    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img


def plot_result_mayank(image_raw, boxes, show=False, save_name=None, color=(255, 0, 0)):
    # plot result
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 3)
    if show:
        # plt.imshow(d_img)
        cv2.imshow("img", d_img)
        # cv2.imshow('img', img)
        ch = cv2.waitKey(4000)
        if ch & 0XFF == ord('q'):
            cv2.destroyAllWindows()
        # cv2.waitKey(1)
        cv2.destroyAllWindows()
    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img

# ## MULTI

def nms_multi(scores, w_array, h_array, thresh_list):
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    # omit not-matching templates
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]
    # extract candidate pixels from scores
    dots = None
    dos_indices = None
    for index, score in zip(indices_omit, scores_omit):
        #here is filtering og score is happening is based on the threshold value*max score value in result matrix
        dot = np.array(np.where(score > thresh_list[index]*score.max()))
        #########
        # i made this changes
        # dots_indices = np.ones(dot.shape[-1]) * index
        if dots is None:
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    dots_indices = dots_indices.astype(np.int)
    x1 = dots[1] - w_array[dots_indices]//2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices]//2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    dots_indices = dots_indices[order]
    
    keep = []
    keep_index = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
        
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2,0,1)
    return boxes, np.array(keep_index)


def plot_result_multi(model,image_raw, boxes, indices, show=False, save_name=None, color_list=None):
    d_img = image_raw.copy()
    if color_list is None:
        color_list = color_palette("hls", indices.max()+1)
        color_list = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), color_list))
    for i in range(len(indices)):
        d_img = plot_result(d_img, boxes[i][None, :,:].copy(), color=color_list[indices[i]])
        # if i>1:
        #     break
        # break

        #################3
        # color_frame
        bbox_info = boxes[i][None, :, :].copy()
        xmin=bbox_info[0, 0][0]
        xmax=bbox_info[0, 1][0]
        ymin=bbox_info[0,0][1]
        ymax=bbox_info[0,1][1]


        # frame = d_img[int(bbox_info[1]):int(bbox_info[3]), int(bbox_info[0]):int(bbox_info[2])]
        frame = d_img[int(ymin):int(ymax), int(xmin):int(xmax)]
        # frame = img[int(bbox_info[1]-15):int(bbox_info[3]+15), int(bbox_info[0]-15):int(bbox_info[2]+15)]
        # frame = img[int(y[value][1]):int(y[value][3]), int(y[value][0]):int(y[value][2])]
        # frame = img[int(y[value][1]-40):int(y[value][3]+40), int(y[value][0]-30):int(y[value][2]+30)]
        cv2.imwrite("data/cust_template/myimage_outPut.jpg", frame)
        # light_col=color_Detect()
        # cv2.putText(d_img, str(light_col), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 0),
        #             lineType=cv2.LINE_AA)
        # print(" time taken in template processing", (time.time() - t1) * 1000)
        #####################################################33
        # Color detector
        ###################################################################333
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        use_cuda = True
        device = torch.device("cuda" if use_cuda else "cpu")
        file_path = '/home/mayank_sati/Desktop/git/2/AI/QATM/data/cust_template/myimage_outPut.jpg'
        img0 = Image.open(file_path)
        img0 = img0.convert("RGB")
        transform = transforms.Compose([transforms.Resize((30, 30)), transforms.ToTensor(), normalize])
        img = transform(img0)
        img = img.unsqueeze(0)
        # input_batch = img0.repeat(ref_batch, 1, 1, 1)
        ############################
        img = img.to(device)
        # t1 = time.time()
        # output = model(img)
        output = model.featex.run_color(img)
        # print("actual time taken", (time.time() - t1) * 1000)
        data = torch.argmax(output, dim=1)
        # print(output)
        traffic_light = ['black', 'green', 'red', 'yellow']
        light_color = traffic_light[int(data)]
        # image = cv2.imread(file_path)
        cv2.putText(d_img, light_color, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
        ############################################################################################
        ####################################################3
    if show:
        # plt.imshow(d_img)
        cv2.imshow("img",d_img)
        # cv2.imshow('img', img)
        ch = cv2.waitKey(20000)
        if ch & 0XFF == ord('q'):
            cv2.destroyAllWindows()
        # cv2.waitKey(1)
        cv2.destroyAllWindows()

    if save_name:
        cv2.imwrite(save_name, d_img[:,:,::-1])
    return d_img


# # RUNNING

def run_one_sample(model, template, image, image_name):
    val = model(template, image, image_name)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)
    
    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i,:,:,0]
        gray = cv2.resize( gray, (image.size()[-1], image.size()[-2]) )

        ###########################33
        # # plt.imshow(d_img)
        # cv2.imshow("imge", gray)
        # # cv2.imshow('img', img)
        # ch = cv2.waitKey(10000)
        # if ch & 0XFF == ord('q'):
        #     cv2.destroyAllWindows()
        # # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # 3
        #############################
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score( gray, w, h) 
        score[score>-1e-7] = score.min()
        score = np.exp(score / (h*w)) # reverse number range back after computing geometry average
        scores.append(score)
    return np.array(scores)


def run_one_sample_mayank(model, dataset):
    for data in dataset:
        # score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        template=data['template']
        image=  data['image']
        image_name= data['image_name']
        w_array=(data['template_w'])
        h_array=(data['template_h'])
        thresh_list=(data['thresh'])

    ######################################
    # cv2.imshow('datasets_frame', image)
    # ch = cv2.waitKey(20000)
    # if ch & 0XFF == ord('q'):
    #     cv2.destroyAllWindows()
    # # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    #####################################
    val = model(template, image, image_name)
    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    val = np.log(val)

    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i, :, :, 0]
        gray = cv2.resize(gray, (image.size()[-1], image.size()[-2]))

        ###########################33
        # # plt.imshow(d_img)
        # cv2.imshow("imge", gray)
        # # cv2.imshow('img', img)
        # ch = cv2.waitKey(10000)
        # if ch & 0XFF == ord('q'):
        #     cv2.destroyAllWindows()
        # # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # 3
        #############################
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score(gray, w, h)
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h * w))  # reverse number range back after computing geometry average
        scores.append(score)
    return np.array(scores),np.array(w_array), np.array(h_array), thresh_list


def run_multi_sample(model, dataset):
    scores = None
    w_array = []
    h_array = []
    thresh_list = []
    for data in dataset:
        score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        if scores is None:
            scores = score
        else:
            scores = np.concatenate([scores, score], axis=0)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.array(scores), np.array(w_array), np.array(h_array), thresh_list
        # break

def run_multi_sample_univ(model, dataset):
    scores = None
    w_array = []
    h_array = []
    thresh_list = []
    for data in dataset:
        score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        if scores is None:
            scores = score
        else:
            scores = np.concatenate([scores, score], axis=0)
        w_array.append(data['template_w'])
        h_array.append(data['template_h'])
        thresh_list.append(data['thresh'])
    return np.array(scores), np.array(w_array), np.array(h_array), thresh_list


if __name__ == '__main__':
    template_dir = 'template/'
    image_path = 'sample/sample1.jpg'
    dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')

    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=True)
    #

    ##################################3333

    # resnet = models.resnet18(pretrained=True)
    # modules = list(resnet.children())[:-1]  # delete the last fc layer.
    # resnet = nn.Sequential(*modules)
    # ### Now set requires_grad to false
    # for param in resnet.parameters():
    #     param.requires_grad = False
    # model = CreateModel(model=resnet, alpha=25, use_cuda=True)
    ########################################

    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)

    boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)

    d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result_sample.png')

    plt.imshow(scores[2])


