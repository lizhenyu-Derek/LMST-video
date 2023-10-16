import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def iou(box1, box2):
    """
    计算iou
    :param box1: 第一个box,格式:(x, y, w, h)
    :param box2: 第一个box,格式:(x, y, w, h)
    :return: iou值
    """
    # convert format
    box1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
    box2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

    # Calculate the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If the intersection is empty, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def get_paths(root, type_):
    files = os.listdir(root)
    paths = []
    for file in files:
        # 若是目录
        if os.path.isdir(os.path.join(root, file)):
            p = get_paths(os.path.join(root, file), type_)
            paths.extend(p)
        else:
            for t in type_:
                if t in file:
                    break
            else:
                continue
            paths.append(os.path.join(root, file))
    return paths
    

def resize_GT(GT, size, to_size):
    GT[0] *= (to_size[0] / size[0])
    GT[1] *= (to_size[1] / size[1])
    GT[2] *= (to_size[0] / size[0])
    GT[3] *= (to_size[1] / size[1])
    GT = [(GT[0] + GT[2]) / 2, (GT[1] + GT[3]) / 2, (GT[2] - GT[0]), (GT[3] - GT[1])]
    return GT


def encode_target(GT_Vascular2, GT_plaque3):
    target = torch.zeros((13, 13, 12))
    if GT_Vascular2:
        cell_x = int(GT_Vascular2[0] / 32)
        cell_y = int(GT_Vascular2[1] / 32)
        # 归一化坐标偏移
        x = (GT_Vascular2[0] % 32) / 32.0
        y = (GT_Vascular2[1] % 32) / 32.0
        # 归一化GT大小
        w = GT_Vascular2[2] / 416.0
        h = GT_Vascular2[3] / 416.0

        target[cell_y][cell_x][:6] = torch.Tensor([x, y, w, h, 1, 0])

    if GT_plaque3:
        cell_x = int(GT_plaque3[0] / 32)
        cell_y = int(GT_plaque3[1] / 32)
        # 归一化坐标偏移
        x = (GT_plaque3[0] % 32) / 32.0
        y = (GT_plaque3[1] % 32) / 32.0
        # 归一化GT大小
        w = GT_plaque3[2] / 416.0
        h = GT_plaque3[3] / 416.0

        target[cell_y][cell_x][6:] = torch.Tensor([x, y, w, h, 1, 1])

    return target


def draw_plt(history, EPOCH, title='Training Loss', path='./plt.png'):
    N = np.arange(0, EPOCH)
    plt.style.use('ggplot')
    plt.figure()
    for n in history:
        plt.plot(N, history[n], label=n)
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss & Accuracy')
    plt.legend()
    plt.savefig(path)


import cv2
def shapen_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.equalizeHist(image)
    return image

if __name__ == '__main__':
    history = {
        'a': [1, 2, 3],
        'b': [4, 5, 6],
    }
    draw_plt(history, 3)