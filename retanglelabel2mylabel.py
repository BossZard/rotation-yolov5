import cv2
import os
import numpy as np
from shapely.geometry import Polygon, MultiPoint  # 多边形
import time
import cv2
import argparse

from time import sleep
def trans(file, line_, wh_list):
    # file = '1001.txt'
    path = label_path + '/' + file

    # line = '' + img_path + '/' + os.path.splitext(file)[0] + '.tif'
    line = ''
    # print(line)
    # print(path)
    f = open(path)
    label = f.read().split()
    # print(label)
    clss = []
    xsets = []
    ysets = []
    sets = []


    for i in range(0, len(label), 9):
        cls = float(label[i]) - 1
        if cls not in clss:
            clss.append(cls)
        data = np.array(label[i+1:i+9]).astype(int)
        data = data.reshape(4, 2)

        rect = cv2.minAreaRect(data)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        # print(rect)
        box = cv2.boxPoints(rect).astype(int)

        c_x = rect[0][0]
        c_y = rect[0][1]
        w = rect[1][0]
        h = rect[1][1]
        theta = rect[-1]


        if (theta < -90 or theta > 0) and h < w:
            print(w,h)
            print(file)
            print(theta)
            sleep(11111)

        if theta == 0 and w < h:
            theta = -90
            t = h
            h = w
            w = t

        if w > h:
            t = h
            h = w
            w = t


        else:
            if theta == 0:
                print('dfasd')
                theta = 0
            else:
                theta = 90 + theta

        if w > h :
            sleep(1111)


        # print(c_x, c_y, w, h, theta)
        # line = line  + ' ' + str(c_x/1024) + ',' + str(c_y/1024) + ',' + str(h / 1024) + ',' + str(w / 1024) + ',' + str(int(theta)) + ',' + str(cls) + ' '
        # line = line + ' ' + str(c_x - h / 2) + ',' + str(c_y - w / 2) + ',' + str(c_x + h / 2) + ',' + str(c_y + w / 2) + ',' + str(cls) + ',' + str(int(theta)+90) + ' '
        line = line + str(cls) + ' ' + str(c_x / 1024) + ' ' + str(c_y / 1024) + ' ' + str(h / 1024) + ' ' + str(w / 1024) + ' ' + str(int(theta)+90) + '\n'
        # line = line + str(cls) + ' ' + str(c_x / 1024) + ' ' + str(c_y / 1024) + ' ' + str(h / 1024) + ' ' + str(
        #     w / 1024) + ' ' + str(int(theta) + 90) + '\n'
        wh_list.append([h/1024, w/1024])
    with open(r'D:\hjj\yolov5\yolov5-ship\convertor\fold0\labels_ship\{}'.format(str(int(os.path.splitext(file)[0])+2008) + '.txt'),
              'w+') as f:

        f.write(line)
    f.close()
    # line_ = line_ + line + '\n'

        # # print(data[:,0].shape)
        # # poly = Polygon(data).convex_hull
        # d_index = np.argmax(data[:, 0])
        # c_index = np.argmax(data[:, 1])
        # c_x = (max(data[:, 0]) + min(data[:, 0])) / 2
        # c_y = (max(data[:, 1]) + min(data[:, 1])) / 2
        # print(data[d_index],data[c_index])
        # # print('len:',len(set(data[:,0])))
        # if len(set(data[:, 0])) not in xsets:
        #     xsets.append(len(set(data[:, 0])))
        # if len(set(data[:, 1])) not in ysets:
        #     ysets.append(len(set(data[:, 1])))
        # if (len(set(data[:, 1]))*len(set(data[:, 0]))) not in sets:
        #     sets.append((len(set(data[:, 1]))*len(set(data[:, 0]))))
        # if len(set(data[:,0])) < 4 or len(set(data[:,1])) < 4:
        #
        #     if len(set(data[:,0])) == 2 and len(set(data[:,1])) == 2:
        #
        #         print('正规矩形：')
        #         theta = - np.pi / 2
        #         right = np.where(data[:, 0]==max(data[:, 0]))
        #         top = np.where(data[:, 1]==max(data[:, 1]))
        #         # print(top[0], right[0])
        #         # h = np.abs(data[top[0][0]][0] - data[top[0][1]][0])
        #         # w = np.abs(data[right[0][0]][1] - data[right[0][1]][1])
        #         #
        #         # print(w , h)
        #     # if len(set(data[:,0])) == 3 or len(set(data[:,1])) == 3:
        #
        #
        #
        # else:
        #     # print(1)
        #     theta = - np.arctan((data[c_index][1] - data[d_index][1]) / (data[d_index][0] - data[c_index][0]))
        #
        #     w = np.sqrt((data[c_index][1] - data[d_index][1])**2 + (data[d_index][0] - data[c_index][0])**2)
        #     h = np.sqrt((data[d_index][0] - data[np.argmin(data[:, 1])][0])**2 +(data[d_index][1] - data[np.argmin(data[:, 1])][1])**2)
        # # print(theta)
        #
        # # print(c_x, c_y, w, h, theta)

    return path, rect, line_, int(theta) + 90, wh_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default=r'D:\hjj\火箭军\科目四按图索骥\科目四初赛第一阶段\train\labels/',
                        help='label path')
    parser.add_argument('--img_path', type=str, default=r'D:\hjj\火箭军\科目四按图索骥\科目四初赛第一阶段\train\images/',
                        help='initial weights path')
    opt = parser.parse_args()
    label_path = opt.label_path
    img_path = opt.img_path
    all_label = []
    cls = []
    xsets = []
    ysets = []
    sets = []
    line_ = ''
    thetas = []
    wh_list = []
    for file in os.listdir(label_path):
        path, ret, line_, theta, wh_list = trans(file, line_, wh_list)
        if theta not in thetas:
            thetas.append(theta)
    print(len(wh_list))
    print(wh_list)
    # print(len(thetas), max(thetas),min(thetas))
    # with open(r'D:\hjj\yolo4/2007_train_ship_angle_1.txt', 'w+') as f:
    #
    #     f.write(line_)
    # f.close()
