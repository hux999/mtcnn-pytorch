import numpy as np
import cv2

from wider_dataset import WiderDataset
import utils


def RndBox(height, width, min_size, max_size):
    size = np.random.randint(min_size, max_size)
    nx = np.random.randint(0, width - size)
    ny = np.random.randint(0, height - size)
    return [nx, ny, nx + size, ny + size]

def GenPNetBoxes(im_size, gt_boxes):
    '''
    input
        im_size: [height, width, channel]
        gt_boxes: [[x1, y1, x2, y2],...]
    '''
    img_h, img_w, _ = im_size
    pos_boxes = []
    neg_boxes = []
    part_boxes = []
    pos_offset = []
    part_offset =[]
    # gen neg sample
    while len(neg_boxes) < 50:
        rnd_box = RndBox(img_h, img_w, 12, min(img_h,img_w)/2)
        iou = utils.IoU(rnd_box, gt_boxes)
        if np.max(iou) < 0.3:
            neg_boxes.append(rnd_box)

    for box in gt_boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if min(w, h) < 20 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        for i in range(5):
            size = np.random.randint(12,  min(img_w, img_h) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            if nx1 + size > img_w or ny1 + size > img_h:
                continue
            rnd_box = [nx1, ny1, nx1 + size, ny1 + size]
            iou = utils.IoU(rnd_box, gt_boxes)

            if np.max(iou) < 0.3:
                # Iou with all gts must below 0.3
                neg_boxes.append(rnd_box)

        # generate positive examples and part faces
        for i in range(20):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > img_w or ny2 > img_h:
                continue
            rnd_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            iou = utils.IoU(rnd_box, box.reshape(1, -1))
            if iou >= 0.65:
                pos_boxes.append(rnd_box)
                pos_offset.append([offset_x1,offset_y1,offset_x2,offset_y2])
            elif iou >= 0.4:
                part_boxes.append(rnd_box)
                part_offset.append([offset_x1,offset_y1,offset_x2,offset_y2])
    return pos_boxes,neg_boxes,part_boxes,pos_offset,part_offset

if __name__ == '__main__':
    dataset = WiderDataset(
            '/home/aldhu/dataset/WIDER/WIDER_train/images/',
            '/home/aldhu/dataset/WIDER/wider_face_split/wider_face_train_bbx_gt.txt')
    num_data = len(dataset)
    print('number of data: %d' % num_data)
    for i in range(num_data):
        img_path,annt = dataset[i]
        im = cv2.imread(img_path)
        gt_boxes = utils.xywh2pts(annt['gt_boxes'])
        pos_boxes,neg_boxes,part_boxes,_,_ = GenPNetBoxes(im.shape, gt_boxes)
        print(len(pos_boxes),len(neg_boxes),len(part_boxes))
        for boxes in pos_boxes:
            x1 = boxes[0]
            y1 = boxes[1]
            x2 = boxes[2]
            y2 = boxes[3]
            cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
        for boxes in neg_boxes:
            x1 = boxes[0]
            y1 = boxes[1]
            x2 = boxes[2]
            y2 = boxes[3]
            cv2.rectangle(im, (x1,y1), (x2,y2), (0,0,255), 2)
        for boxes in part_boxes:
            x1 = boxes[0]
            y1 = boxes[1]
            x2 = boxes[2]
            y2 = boxes[3]
            cv2.rectangle(im, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.imshow('im', im)
        cv2.waitKey()

