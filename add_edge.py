import cv2
import os
# from keras.application.vgg import VGG16
import matplotlib.pyplot as plt
if __name__ == '__main__':
    label_path = r'convertor\fold0\labels\train2017/'
    img_path = r'inference\test2'
    edge_size = 100

    for file in os.listdir(img_path):

        # print(os.path.join(img_path, (file.split('.')[0] + '.tif')))
        img = cv2.imread(os.path.join(img_path, (file.split('.')[0] + '.tif')))



        img  = cv2.copyMakeBorder(img,edge_size,edge_size,edge_size,edge_size, cv2.BORDER_CONSTANT,value=[144,144,144])
        cv2.imwrite(r'G:\hjj\yolov5\yolov5-ship\inference\edge_pic/{}'.format(file), img)

        # plt.imshow(img)
        # plt.show()
        # print(img)