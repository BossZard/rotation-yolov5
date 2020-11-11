import numpy as np
import xml.etree.ElementTree as ET
import glob
import random
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
    path = opt.label_path + '/' + file

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
    # with open(r'D:\hjj\yolov5-master\convertor\fold0\labels\theta\{}'.format(os.path.splitext(file)[0] + '.txt'),
    #           'w+') as f:
    #
    #     f.write(line)
    # f.close()
    line_ = line_ + line + '\n'

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




def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(box,k):
    # 取出一共有多少框
    row = box.shape[0]
    
    # 每个框各个点的位置
    distance = np.empty((row,k))
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # 取出最小点
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default=r'D:\hjj\火箭军\科目四按图索骥\科目四初赛第一阶段\train\labels/', help='label path')
    opt = parser.parse_args()
    # label_path = r'D:\hjj\火箭军\科目四按图索骥\科目四初赛第一阶段\train\labels/'

    all_label = []
    cls = []
    xsets = []
    ysets = []
    sets = []
    line_ = ''
    thetas = []
    wh_list = []
    for file in os.listdir(opt.label_path):
        path, ret, line_, theta, wh_list = trans(file, line_, wh_list)
        if theta not in thetas:
            thetas.append(theta)

    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    SIZE = 1024
    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    path = r'./VOCdevkit/VOC2007/Annotations'
    
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    # data = load_data(path)
    data = np.array(wh_list)
    
    # 使用k聚类算法
    out = kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()