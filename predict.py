import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import time
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import argparse


model_type = 'small'


class Dnetwork(nn.Module):
    def __init__(self, CNN_hidden_size):
        super(Dnetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc_bbox = nn.Linear(16 * 16 * 64, CNN_hidden_size)
        self.fc_bbox2 = nn.Linear(CNN_hidden_size, 4)
        self.dropout = nn.Dropout(p=0.35)
        self.fc_label = nn.Linear(16 * 16 * 64, CNN_hidden_size)
        self.fc_label2 = nn.Linear(CNN_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input):
        CNN_output = self.layer1(input)
        CNN_output = self.layer2(CNN_output)
        CNN_output = self.layer3(CNN_output)
        CNN_output = CNN_output.view(CNN_output.size(0), -1)
        CNN_output = self.dropout(CNN_output)
        bbox_output = self.fc_bbox(CNN_output)
        bbox_output = self.relu(bbox_output)
        bbox_output = self.fc_bbox2(bbox_output)
        label_output = self.fc_label(CNN_output)
        label_output = self.relu(label_output)
        label_output = torch.squeeze(self.sigmoid(self.fc_label2(label_output)))
        return bbox_output, label_output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeeperNN(nn.Module):
    def __init__(self):
        super(DeeperNN, self).__init__()
        hidden_size = 1000
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(3, 16)
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(16, 32)
        self.bn2 = norm_layer(32)
        self.max_pool = nn.MaxPool2d(2)
        self.conv3 = conv1x1(32, 64)
        self.bn3 = norm_layer(64)
        self.conv4 = conv3x3(64, 64)
        self.bn4 = norm_layer(64)
        self.conv5 = conv1x1(64, 32)
        self.bn5 = norm_layer(32)
        s = 100352
        self.fc_label = nn.Linear(s, hidden_size)
        self.fc_label2 = nn.Linear(hidden_size, 1)
        self.fc_bbox = nn.Linear(s, hidden_size)
        self.fc_bbox2 = nn.Linear(hidden_size, 4)
        self.dropout = nn.Dropout(p=0.35)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.max_pool(out)
        for i in range(10):
            identity = out
            out = self.conv3(out)
            out = self.bn3(out)
            out = self.relu(out)

            out = self.conv4(out)
            out = self.bn4(out)
            out = self.relu(out)

            out = self.conv5(out)
            out = self.bn5(out)

            out += identity
            out = self.relu(out)
        representation = out.view(out.size(0), -1)
        representation = self.dropout(representation)
        out_bbox = self.fc_bbox(representation)
        out_bbox = self.relu(out_bbox)
        out_bbox = self.fc_bbox2(out_bbox)
        out_labels = self.fc_label(representation)
        out_labels = self.relu(out_labels)
        out_labels = torch.squeeze(self.sigmoid(self.fc_label2(out_labels)))
        return out_bbox, out_labels


class ResNetLike(nn.Module):
    def __init__(self):
        super(ResNetLike, self).__init__()
        self.conv7x7 = nn.Conv2d(3, 64, 7, 2, 5)
        self.normalization1 = nn.BatchNorm2d(64)
        self.pooling = nn.MaxPool2d(3, 2, 1)
        self.conv3x3_64 = list()
        for i in range(6):
            self.conv3x3_64.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 'same'), nn.BatchNorm2d(64), ))
        self.conv3x3_128_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.normalization2 = nn.BatchNorm2d(128)
        self.down_sample1 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2),
        nn.BatchNorm2d(128),
        )
        self.conv3x3_128 = list()
        for i in range(7):
            self.conv3x3_128.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 'same'), nn.BatchNorm2d(128),))
        self.conv3x3_256_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.normalization3 = nn.BatchNorm2d(256)
        self.down_sample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),
        )
        self.conv3x3_256 = list()
        for i in range(11):
            self.conv3x3_256.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 'same'), nn.BatchNorm2d(256), ))
        self.conv3x3_512_1 = nn.Conv2d(256, 512, 3, 2, 1)
        self.normalization4 = nn.BatchNorm2d(512)
        self.down_sample3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2),
            nn.BatchNorm2d(512),
        )
        self.conv3x3_512 = list()
        for i in range(5):
            self.conv3x3_512.append(nn.Sequential(nn.Conv2d(512, 512, 3, 1, 'same'), nn.BatchNorm2d(512)))
        self.avg_pooling = nn.AvgPool2d(3, 2, 1)
        self.relu = nn.ReLU()
        size = 4 * 4
        hidden_size = 1000
        self.fc_label = nn.Linear(512 * size, hidden_size)
        self.fc_label2 = nn.Linear(hidden_size, 1)
        self.fc_bbox = nn.Linear(512 * size, hidden_size)
        self.fc_bbox2 = nn.Linear(hidden_size, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.conv7x7(image)
        x = self.normalization1(x)
        x = self.pooling(x)
        identity = x
        for i in range(3):
            x = self.relu(x)
            x = self.conv3x3_64[2*i](x)
            x = self.relu(x)
            x = self.conv3x3_64[2*i+1](x)
            x = x + identity
            identity = x
        x = self.relu(x)
        identity = self.down_sample1(identity)
        x = self.conv3x3_128_1(x)
        x = self.normalization2(x)
        x = self.relu(x)
        x = self.conv3x3_128[0](x)
        x = x + identity
        identity = x
        for i in range(3):
            x = self.relu(x)
            x = self.conv3x3_128[2*i + 1](x)
            x = self.relu(x)
            x = self.conv3x3_128[2*i + 2](x)
            x = x + identity
            identity = x
        identity = self.down_sample2(identity)
        x = self.relu(x)
        x = self.conv3x3_256_1(x)
        x = self.normalization3(x)
        x = self.relu(x)
        x = self.conv3x3_256[0](x)
        x = x + identity
        for i in range(5):
            x = self.relu(x)
            x = self.conv3x3_256[2*i + 1](x)
            x = self.relu(x)
            x = self.conv3x3_256[2*i + 2](x)
            x = x + identity
            identity = x
        identity = self.down_sample3(identity)
        x = self.relu(x)
        x = self.conv3x3_512_1(x)
        x = self.normalization4(x)
        x = self.relu(x)
        x = self.conv3x3_512[0](x)
        x = x + identity
        for i in range(2):
            x = self.relu(x)
            x = self.conv3x3_512[2 * i + 1](x)
            x = self.relu(x)
            x = self.conv3x3_512[2 * i + 2](x)
            x = x + identity
            identity = x
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        out_labels = self.fc_label(x)
        out_labels = self.relu(out_labels)
        out_labels = torch.squeeze(self.sigmoid(self.fc_label2(out_labels)))
        out_bbox = self.fc_bbox(x)
        out_bbox = self.relu(out_bbox)
        out_bbox = self.fc_bbox2(out_bbox)
        return out_bbox, out_labels


def test(test_loader, netD):
    netD.eval()
    outputs_bboxes = []
    outputs_labels_list = []
    for data in test_loader:
        if torch.cuda.is_available():
            data.cuda()
        images = data[0]

        outputs_bbox, outputs_labels = netD(images)
        outputs_bboxes.append(outputs_bbox.detach().numpy())
        outputs_labels_list.append(outputs_labels.detach().numpy())
    pred_bboxes = np.concatenate(outputs_bboxes)
    pred_labels = np.round(np.concatenate(outputs_labels_list)).astype(np.bool)
    return pred_bboxes, pred_labels


class Tuple_DataSet(Dataset):
    def __init__(self, image_dir):
        super(Tuple_DataSet, self).__init__()
        self.image_dir = image_dir
        self.bboxes = []
        self.proper_mask_indicators = []
        self.file_names = []
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), ])
        example_filenames = os.listdir(image_dir)
        for filename in example_filenames:
            image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
            bbox = json.loads(bbox)
            bbox = torch.tensor(bbox)
            if bbox[2] < 0 or bbox[3] < 0:
                continue
            self.bboxes.append(bbox)
            self.proper_mask_indicators.append(float(1.0) if proper_mask.lower() == "true" else float(0.0))
            self.file_names.append(filename)

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        im = Image.open(self.image_dir +'/' + self.file_names[idx]).convert('RGB')
        im = self.trsfm(im)
        im = F.pad(input=im, pad=(0, 224 - im.shape[2], 0, 224 - im.shape[1], 0, 0), mode='constant', value=0)
        if model_type == 'small':
            trans_to_image = transforms.ToPILImage()
            im = trans_to_image(im).resize((128, 128))
            trans_to_tensor = transforms.ToTensor()
            im = trans_to_tensor(im)
        return im, self.bboxes[idx], self.proper_mask_indicators[idx]


if __name__ == '__main__':

    batch_size = 200
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    # Reading input folder
    files = os.listdir(args.input_folder)
    test_dataset = Tuple_DataSet(args.input_folder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if model_type == 'small':
        with open('netD', 'rb') as f:
            netD = pickle.load(f)
    else:
        with open('netDt4', 'rb') as f:
            netD = pickle.load(f)

    bbox_pred, proper_mask_pred = test(test_loader, netD)
    prediction_df = pd.DataFrame(zip(files, *(bbox_pred.T), proper_mask_pred),
                                 columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
    prediction_df.to_csv("prediction.csv", index=False, header=True)
