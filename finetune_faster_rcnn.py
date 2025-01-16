import os
import warnings
from xml.etree import ElementTree as et

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils

# removes warning 'iccp known incorrect srgb profile'
warnings.filterwarnings('ignore')


# chess peice dataset class that inherits from pytorch's dataset class
class ChessPieceImageDataset(torch.utils.data.Dataset):

    def __init__(self, train_data, width, height, transforms=None):
        self.transforms = transforms
        self.train_data = train_data
        self.height = height
        self.width = width

        # define training images as files that end in .jpg
        self.images = [image for image in sorted(os.listdir(train_data)) if image[-4:] == '.jpg']

        # define possible classes for classification
        # 0th class is reserved for background
        self.classes = [None,
                        'wpawn',
                        'bpawn',
                        'wrook',
                        'brook',
                        'wknight',
                        'bknight',
                        'wbishop',
                        'bbishop',
                        'wqueen',
                        'bqueen',
                        'wking',
                        'bking']

    # must overide __getitem__
    # allows for indexed retrival of dataset ([])
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.train_data, image_name)

        # recolor
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # resize
        # inter_area is typically used for shrinking image
        image_resized = cv2.resize(image_rgb, (self.width, self.height), cv2.INTER_AREA)

        # ensures pixel values are not out of rgb bounds after resizing
        image_resized /= 255.0

        # annotations and annotations path
        annotation_file = image_name[:-4] + '.xml'
        annotation_file_path = os.path.join(self.train_data, annotation_file)

        # bounding boxes
        boxes = []

        # labels
        labels = []

        # xml reader
        tree = et.parse(annotation_file_path)
        root = tree.getroot()

        # cv2 image height and width
        wt = image.shape[1]
        ht = image.shape[0]

        # loop through all image annotations and define bounding boxes and labels
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # transform bounding boxes to tensors for gpu processing
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # get area of bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # set to not crowded for simplicity
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # target is function return value
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd

        # image id is its index coverted to a tensor for gpu processing
        image_id = torch.tensor([index])

        target["image_id"] = image_id

        # applies transforms to image if transforms are set
        if self.transforms:

            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=target['labels'])

            # sets image return value to resized image
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)

        return image_resized, target

    # returns length of dataset
    def __len__(self):
        return len(self.images)


# model
def get_object_detection_model(num_classes):
    # load a model pre-trained on COCO image set
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# set train=True for training transforms and False for validation/test transforms
def get_transform(train):
    if train:
        return A.Compose([A.HorizontalFlip(0.5), ToTensorV2(p=1.0)],
                         bbox_params={'format': 'pascal_voc',
                                      'label_fields': ['labels']})
    else:
        return A.Compose([ToTensorV2(p=1.0)],
                         bbox_params={'format': 'pascal_voc',
                                      'label_fields': ['labels']})


if __name__ == '__main__':
    # train and test image data
    train_data = 'images/train'

    # use our dataset and defined transformations
    dataset = ChessPieceImageDataset(train_data, 480, 480, transforms=get_transform(True))
    dataset_test = ChessPieceImageDataset(train_data, 480, 480, transforms=get_transform(False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    tsize = int(len(dataset)*test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define number of classes for classification
    num_classes = 13

    # create model usingt helper function or load saved model
    if os.path.exists('saved_model/saved_model.pth'):
        print('Loading saved model...')
        model = torch.load('saved_model/saved_model.pth')
    else:
        print('Creating new model...')
        model = get_object_detection_model(num_classes)

    # move model to the correct device (gpu)
    model.to(device)

    # construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

    # construct learning rate scheduler which decreases the learning rate by 30% every 100 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)

    # training for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):

        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        if epoch % 15 == 0:
            evaluate(model, data_loader_test, device=device)

        if epoch % 25 == 0:
            torch.save(model, 'saved_model/saved_model.pth')

        # reset gradients
        optimizer.zero_grad()

    torch.save(model, 'saved_model/saved_model.pth')
    # torch.save(model.state_dict(), 'inference/inference.pth')
