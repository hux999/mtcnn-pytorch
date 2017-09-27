import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import cv2

from mtcnn_model import PNet,LossFn
from mtcnn_data import GenPNetBoxes
from wider_dataset import WiderDataset
import utils

class PNetDataset(Dataset):
    ''' MTCNN Dataset '''
    def __init__(self, wider_face):
        self.wider_face = wider_face

    def __len__(self):
        return len(self.wider_face)

    def __getitem__(self, index):
        img_path,annt = self.wider_face[index]
        im = cv2.imread(img_path)
        gt_boxes = utils.xywh2pts(annt['gt_boxes'])
        pos_boxes,neg_boxes,part_boxes,pos_offset,part_offset = \
                GenPNetBoxes(im.shape, gt_boxes)
        num_pos = len(pos_boxes)
        num_neg = min(len(neg_boxes), num_pos*2)
        num_part = min(len(part_boxes), num_pos)
        crop_list = []
        offset_list = []
        class_list = []
        mask_list = []

        # pos sample
        trans = transforms.ToTensor()
        for (x1,y1,x2,y2),offset in zip(pos_boxes,pos_offset):
            crop_im = cv2.resize(im[y1:y2, x1:x2, :], (12,12), \
                    interpolation=cv2.INTER_LINEAR)
            crop_list.append(trans(crop_im))
            class_list.append(1)
            offset_list.append(offset)
            mask_list.append([1,1,0])

        # neg sample
        if num_neg > len(neg_boxes):
            rnd_idx = np.random.permutation(len(neg_boxes))[:num_neg]
        else:
            rnd_idx = range(len(neg_boxes))
        for i in rnd_idx:
            x1,y1,x2,y2 = neg_boxes[i]
            crop_im = cv2.resize(im[y1:y2, x1:x2, :], (12,12), \
                    interpolation=cv2.INTER_LINEAR)
            crop_list.append(trans(crop_im))
            class_list.append(0)
            offset_list.append([0,0,0,0])
            mask_list.append([1,0,0])

        # part sample
        if num_part > len(part_boxes):
            rnd_idx = np.random.permutation(len(part_boxes))[:num_part]
        else:
            rnd_idx = range(len(part_boxes))
        for i in rnd_idx:
            x1,y1,x2,y2 = part_boxes[i]
            offset = part_offset[i]
            crop_im = cv2.resize(im[y1:y2, x1:x2, :], (12,12), \
                    interpolation=cv2.INTER_LINEAR)
            crop_list.append(trans(crop_im))
            class_list.append(0) 
            offset_list.append(offset)
            mask_list.append([0,1,0])

        # combine
        imgdata = torch.stack(crop_list)
        label = torch.LongTensor(class_list)
        offset = torch.Tensor(offset_list)
        landmark = torch.zeros(len(crop_list),10)
        beta = torch.LongTensor(mask_list)
        return imgdata,label,offset,landmark,beta


class CollateFn:

    def __init__(self):
        pass

    def __call__(self, batch_list):
        img_list = []
        label_list = []
        offset_list = []
        landmark_list = []
        beta_list = []
        for batch_data in batch_list:
            imgdata,label,offset,landmark,beta = batch_data
            img_list.append(imgdata)
            label_list.append(label)
            offset_list.append(offset)
            landmark_list.append(landmark)
            beta_list.append(beta)
        batch_img = torch.cat(img_list)
        batch_label = torch.cat(label_list)
        batch_offset = torch.cat(offset_list)
        batch_landmark = torch.cat(landmark_list)
        batch_beta = torch.cat(beta_list)
        return batch_img,batch_label,batch_offset,batch_landmark,batch_beta

def Train(train_data, net, num_epoch=10, lr=0.001, use_cuda=False):
    if use_cuda:
        net.cuda()
        net_ = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5])
    else:
        net_ = net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = LossFn([1.0,0.5,0.5])
    for i_epoch in range(num_epoch):
        # train
        net_.train()
        batch_data = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8,
                collate_fn = CollateFn())
        for i_batch, (img,gt_label,gt_offset,gt_landmark,beta) in enumerate(batch_data):
            img = Variable(img)
            gt_label = Variable(gt_label)
            gt_offset = Variable(gt_offset)
            gt_landmark = Variable(gt_landmark)
            if use_cuda:
                img = img.cuda()
                gt_label = gt_label.cuda()
                gt_offset = gt_offset.cuda()
                gt_landmark = gt_landmark.cuda()
                beta = beta.cuda()
            # forward
            pred_label,pred_offset,pred_landmark = net_(img)
            loss = loss_fn.loss(gt_label, gt_offset, gt_landmark, beta,
                    pred_label, pred_offset, pred_landmark)
            if i_batch % 10 == 0:
                print('epoch:%d, batch:%s, loss:%f' % (i_epoch, i_batch, loss.data[0]))
                print('num_boxes:%d' % pred_label.size()[0])
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # save model for each epoch
        torch.save(net.state_dict(), ('epoch_%d.pt' % i_epoch))

if __name__ == '__main__':
    wider_face_train = WiderDataset(
            '/home/aldhu/dataset/WIDER/WIDER_train/images/',
            '/home/aldhu/dataset/WIDER/wider_face_split/wider_face_train_bbx_gt.txt')
    print('num_data:%d' % len(wider_face_train))
    pnet_data = PNetDataset(wider_face_train)
    
    net = PNet(is_train=True)

    Train(pnet_data, net, 30, 0.001, use_cuda=True) 
    Train(pnet_data, net, 30, 0.0001, use_cuda=True) 
