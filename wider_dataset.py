import sys
import os

import numpy as np
import cv2

class WiderDataset:
    ''' WIDER FACE dataset'''
    def __init__(self, img_root, annt_file):
        self.img_root = img_root
        self.annt_list = self._load_annotation(annt_file)

    def _load_annotation(self,filename):
        annt_list = []
        with open(filename) as fin:
            while True:
                img_path = fin.readline().strip()
                if img_path == '':
                    break
                num_faces = int(fin.readline().strip())
                annt = {}
                annt['gt_boxes'] = []
                annt['blur'] = []
                annt['expression'] = []
                annt['illumination'] = []
                annt['invalid'] = []
                annt['occlusion'] = []
                annt['pose'] = []
                for i in range(num_faces):
                    val = [ int(v) for v in fin.readline().strip().split() ]
                    annt['gt_boxes'].append(val[:4])
                    annt['blur'].append(val[4])
                    annt['expression'].append(val[5])
                    annt['illumination'].append(val[6])
                    annt['invalid'].append(val[7])
                    annt['occlusion'].append(val[8])
                    annt['pose'].append(val[9])
                for key in annt.keys():
                    annt[key] = np.array(annt[key])
                annt_list.append((img_path,annt))
        return annt_list

    def __len__(self):
        return len(self.annt_list)

    def __getitem__(self, index):
        img_path,annt = self.annt_list[index]
        img_path = os.path.join(self.img_root, img_path)
        return img_path,annt

if __name__ == '__main__':
    dataset = WiderDataset(
            '/home/aldhu/dataset/WIDER/WIDER_train/images',
            '/home/aldhu/dataset/WIDER/wider_face_split/wider_face_train_bbx_gt.txt')

    num_data = len(dataset)
    for i in range(num_data):
        img_path,annt = dataset[i]
        im = cv2.imread(img_path)
        gt_boxes = annt['gt_boxes']
        for i in range(gt_boxes.shape[0]):
            x = gt_boxes[i,0]
            y = gt_boxes[i,1]
            w = gt_boxes[i,2]
            h = gt_boxes[i,3]
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow('im', im)
        cv2.waitKey()
