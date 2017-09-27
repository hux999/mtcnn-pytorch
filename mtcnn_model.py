import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        #nn.init.xavier_uniform(m.bias.data)  

class PNet(nn.Module):
    ''' PNet '''
    def __init__(self, is_train=False):
        super(PNet, self).__init__()
        self.is_train = is_train
        # backend
        self.feature = nn.Sequential(
                nn.Conv2d(3, 10, kernel_size=3, stride=1), # conv1
                nn.PReLU(), # PReLU1
                nn.MaxPool2d(kernel_size=2, stride=2), # pool1
                nn.Conv2d(10, 16, kernel_size=3, stride=1), # conv2
                nn.PReLU(), # PReLU2
                nn.Conv2d(16, 32, kernel_size=3, stride=1), # conv3
                nn.PReLU() # PReLU3
                )
        # detection
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.feature(x)
        label = self.conv4_1(x)
        if self.is_train is False:
            label = F.softmax(label)
        offset = self.conv4_2(x)
        landmark = self.conv4_3(x)
        return label,offset,landmark

class LossFn:
    def __init__(self, alpha):
        # loss function
        self.alpha = alpha
        self.loss_det = nn.CrossEntropyLoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    def loss(self, gt_label, gt_offset, gt_landmark, beta,
            pred_label, pred_offset, pred_landmark):
        pred_label = torch.squeeze(pred_label)
        idx_det = torch.nonzero(beta[:,0]).squeeze()
        loss_det = self.loss_det(pred_label[idx_det,:], gt_label[idx_det])
        idx_box = torch.nonzero(beta[:,1]).squeeze()
        loss_box = self.loss_box(pred_offset[idx_box,:], gt_offset[idx_box,:])
        '''
        idx_landmark = torch.nonzero(beta[:,2]).squeeze()
        loss_landmark = self.loss_landmark(
                pred_landmark[idx_landmark,:],gt_landmark[idx_landmark, :])
        return self.alpha[0]*loss_det + self.alpha[1]*loss_box + \
                self.alpha[2]*loss_landmark
        '''
        return self.alpha[0]*loss_det + self.alpha[1]*loss_box

class RNet(nn.Module):
    ''' RNet '''
    def __init__(self):
        super(RNet, self).__init__()
        # backend
        self.feature = nn.Sequential(
                nn.Conv2d(3, 28, kernel_size=3, stride=1), # conv1
                nn.PReLU(), # prelu1
                nn.MaxPool2d(kernel_size=3, stride=2), # pool1
                nn.Conv2d(28, 48, kernel_size=3, stride=1), # conv2
                nn.PReLU(), # prelu2
                nn.MaxPool2d(kernel_size=3, stride=2), # pool2
                nn.Conv2d(48, 64, kernel_size=2, stride=1), # conv3
                nn.PReLU() # prelu3
                )
        self.conv4 = nn.Linear(64,128) # conv4
        self.prelu4 = nn.PReLU() # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 2)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.feature(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = self.conv5_1(x)
        box = self.conv5_2(x)
        landmard = self.conv5_3(x)
        return det,box,landmark
        
if __name__ == '__main__':
    net = PNet()
    x = Variable(torch.rand(1,3,12,12))
    y1,y2,y3 = net(x)
    print(y1)
    print(y2)
    print(y3)
