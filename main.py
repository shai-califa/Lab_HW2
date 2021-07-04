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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image

train_dir = "traintest/train/"
test_dir = "traintest/test/"
image_dir = train_dir


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


def parse_images_and_bboxes(image_dir):
    """
    Parse a directory with images.
    :param image_dir: Path to directory with images.
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    for filename in example_filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        print(bbox[0])
        bbox = json.loads(bbox)
        proper_mask = True if proper_mask.lower() == "true" else False
        data.append((filename, image_id, bbox, proper_mask))
    return data


def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union


def calc_iou_lists(list_a, list_b):
  iou = 0.0
  for it1, it2 in zip(list_a, list_b):
    iou += calc_iou(it1, it2)
  return iou/len(list_a)


def show_images_and_bboxes(data, image_dir):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    for filename, image_id, bbox, proper_mask in data:
        # Load image
        im = cv2.imread(os.path.join(image_dir, filename))
        # BGR to RGB
        im = im[:, :, ::-1]
        # Ground truth bbox
        x1, y1, w1, h1 = bbox
        # Predicted bbox
        x2, y2, w2, h2 = random_bbox_predict(bbox)
        # Calculate IoU
        iou = calc_iou(bbox, (x2, y2, w2, h2))
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        rect = patches.Rectangle((x2, y2), w2, h2,
                                 linewidth=2, edgecolor='b', facecolor='none', label='predicted')
        ax.add_patch(rect)
        fig.suptitle(f"proper_mask={proper_mask}, IoU={iou:.2f}")
        ax.axis('off')
        fig.legend()
        plt.show()
        print(im.shape)


def calculate_label_accuracy(list_label_a, list_label_b):
    counter = sum([1 for a, b in zip(list_label_a, list_label_b) if round(a) == round(b)])
    return counter/len(list_label_b)


def random_bbox_predict(bbox):
    """
    Randomly predicts a bounding box given a ground truth bounding box.
    For example purposes only.
    :param bbox: Iterable with numbers.
    :return: Random bounding box, relative to the input bbox.
    """
    return [x + np.random.randint(-15, 15) for x in bbox]


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
        im = Image.open(self.image_dir + self.file_names[idx]).convert('RGB')
        im = self.trsfm(im)
        im = F.pad(input=im, pad=(0, 224 - im.shape[2], 0, 224 - im.shape[1], 0, 0), mode='constant', value=0)
        # trans_to_image = transforms.ToPILImage()
        # im = trans_to_image(im).resize((128, 128))
        # trans_to_tensor = transforms.ToTensor()
        # im = trans_to_tensor(im)
        return im, self.bboxes[idx], self.proper_mask_indicators[idx]


def plotter_epoch_progress(name, res_list):
    res_list = res_list[1:]
    epochs = list(range(len(res_list)))
    plt.figure()
    plt.title(name)
    plt.plot(epochs, res_list)


def test(test_loader, netD, bce_criterion, mse_criterion):
    netD.eval()
    iou = 0
    accuracy = 0
    running_loss = 0
    for data in test_loader:
        if torch.cuda.is_available():
            data.cuda()
        images = data[0]
        bboxes = data[1]
        labels = data[2]
        labels = labels.float()
        bboxes = bboxes.float()

        outputs_bbox, outputs_labels = netD(images)
        loss = bce_criterion(outputs_labels, labels) + mse_criterion(outputs_bbox, bboxes)
        running_loss += loss.item()
        n = len(outputs_bbox.tolist())
        iou += calc_iou_lists(outputs_bbox.tolist(), bboxes.tolist()) * n
        accuracy += calculate_label_accuracy(outputs_labels.tolist(), labels.tolist()) * n
    n = test_dataset.__len__()
    print(f'test loss: {running_loss / n}')
    print(f'test accuracy: {accuracy / n}')
    print(f'test iou: {iou / n}')
    netD.train()
    return accuracy/n, iou/n, running_loss / n


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    batch_size = 200
    train_dataset = Tuple_DataSet(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Tuple_DataSet(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # netD = Dnetwork(1000)
    # netD = DeeperNN()
    netD = ResNetLike()
    netD.apply(weights_init)

    if torch.cuda.is_available():
        netD.cuda()

    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    epochs = 20
    D_losses = []
    D_accuracy = []
    D_iou = []
    D_test_loss = []
    timeElapsed = []
    running_loss = 0.0
    for epoch in range(epochs):
        print("# Starting epoch [%d/%d]..." % (epoch, epochs))
        for i, data in enumerate(train_loader):
            if torch.cuda.is_available():
                data.cuda()
            images = data[0]
            bboxes = data[1]
            labels = data[2]
            labels = labels.float()
            bboxes = bboxes.float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs_bbox, outputs_labels = netD(images)
            loss = bce_criterion(outputs_labels, labels) + mse_criterion(outputs_bbox, bboxes)
            running_loss += loss.item()
            if epoch % 2 == 0 and i == 0:  # print every 5 epochs
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (2*train_dataset.__len__())))
                accuracy, iou, test_loss = test(test_loader, netD, bce_criterion, mse_criterion)
                D_losses.append(running_loss / (2*train_dataset.__len__()))
                D_iou.append(iou)
                D_accuracy.append(accuracy)
                D_test_loss.append(test_loss)
                running_loss = 0.0
            loss.backward()
            optimizer.step()
        print("# Finished epoch.")
    print("# Finished.")

    with open('D_lossest4', 'wb') as f:
        pickle.dump(D_losses, f)
    with open('D_iout4', 'wb') as f:
        pickle.dump(D_iou, f)
    with open('D_accuracyt4', 'wb') as f:
        pickle.dump(D_accuracy, f)
    with open('netDt4', 'wb') as f:
        pickle.dump(netD, f)
    with open('D_test_losst4', 'wb') as f:
        pickle.dump(D_test_loss, f)

    with open('D_lossest4', 'rb') as f:
        D_losses = pickle.load(f)
    with open('D_iout4', 'rb') as f:
        D_iou = pickle.load(f)
    with open('D_accuracyt4', 'rb') as f:
        D_accuracy = pickle.load(f)
    with open('netDt4', 'rb') as f:
        netD = pickle.load(f)
    with open('D_test_losst4', 'rb') as f:
        D_test_loss = pickle.load(f)

    plotter_epoch_progress('loss per 2 epochs', D_losses)
    plotter_epoch_progress('test loss per 2 epochs', D_test_loss)
    plotter_epoch_progress('test iou per 2 epochs', D_iou)
    plotter_epoch_progress('test accuracy per 2 epochs', D_accuracy)
    plt.show()
    test(test_loader, netD, bce_criterion, mse_criterion)

