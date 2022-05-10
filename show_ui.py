# -*- coding: utf-8 -*-
"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:24/12/2021
"""


from ui.final_version import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor
import sys
import os
import numpy as np
import torch
import cv2
import torchvision
import time
from PyQt5.QtWidgets import QFileDialog, QLabel, QListView
from PyQt5.QtGui import QPixmap, QImage, QIcon, QTextCursor
from PyQt5.QtCore import QSize
from utils.gradcam import *
from utils.single_img_predict import *
from utils.kNN_retrival import *

from utils.read_dataset_img import *
from utils.load_pretrain_model import *
from utils.all_dataset_reference import *

import cgitb
cgitb.enable(format='text')   # output the error message


class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.image_name = ''
        self.cub_images = read_cub_img()
        self.car_images = read_car_img()
        self.aircraft_images = read_aircraft_img()
        # self.thread = Thread()
        # self.pushButton_10.clicked.connect(self.slotStart)
        # self.thread.sinout.connect(self.slotAdd)

    # def slotStart(self):
    #     self.thread.start()
    # def slotAdd(self, file_str):
    #     self.textBrowser.append(file_str)

    def select_image(self):
        if self.radioButton_4.isChecked() == True:  # 选中了 cub 数据集
            # self.viewList(self.cub_images)
            self.image_name, _ = QFileDialog.getOpenFileName(self, '选择图片', 'G:\\github_files\\datasets\\CUB\\', 'Image files(*.jpg *.gif *.png)')

        elif self.radioButton_5.isChecked() == True:  # 选中了 car 数据集
            # self.viewList(self.car_images)
            self.image_name, _ = QFileDialog.getOpenFileName(self, '选择图片', 'G:\\github_files\\datasets\\CAR\\', 'Image files(*.jpg *.gif *.png)')

        elif self.radioButton_6.isChecked() == True:  # 选中了 aircraft
            # self.viewList(self.aircraft_images)
            self.image_name, _ = QFileDialog.getOpenFileName(self, '选择图片', 'G:\\github_files\\datasets\\Aircraft\\', 'Image files(*.jpg *.gif *.png)')
        print(self.image_name)


        if os.path.exists(self.image_name):
            pic = QPixmap(self.image_name)
            self.label_2.setPixmap(pic)
            self.label_2.setScaledContents(True)
        try:
            self.listWidget.setCurrentRow(0)
        except Exception as inst:
            print(inst)

    def viewList(self, images):
        self.textBrowser_1.clear()
        for i in images:
            self.textBrowser_1.append(i)
        try:
            self.textBrowser_1.setCurrentRow(0)
            self.suoluetu()
        except Exception as inst:
            print(inst)

    def show_img(self):
        start = time.time()

        dataset_name = self.image_name.split('/')[-4]  # get the name of dataset
        model = load_model(dataset_name)
        # grad-cam
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=False)
        img = cv2.imread(self.image_name, 1)
        cv2.imwrite("F:\\chapter_3_heatmap\\init.jpg", img)
        img = np.float32(img) / 255
        # Opencv loads as BGR
        img = img[:, :, ::-1]
        input_img = preprocess_image(img)
        target_category = None
        grayscale_cam = grad_cam(input_img, target_category)
        grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
        pic = show_cam_on_image(img, grayscale_cam)
        cv2.imwrite("F:\\chapter_3_heatmap\\ad.jpg", pic)
        # print(pic.shape)

        cam = QImage(pic[:], pic.shape[1], pic.shape[0], pic.shape[1] * 3, QImage.Format_RGB888)  # 转化为QImage
        pic = QPixmap(cam).scaled(pic.shape[1], pic.shape[0])  # 设置图片大小
        self.label_3.setPixmap(pic)
        self.label_3.setScaledContents(True)

        pred_class, pred_con = eval_single_img(self.image_name, model)
        print(pred_class, pred_con, time.time() - start)
        print('self.image_name.split("/")[-2].split("/")[-1]', self.image_name.split("/")[-2].split(".")[-1])
        self.lineEdit_7.setText(pred_class)
        # self.lineEdit_12.setText(self.image_name.split('/')[-2].split('.')[-1])  # 展示真实类别
        self.lineEdit_12.setText(str(format(pred_con * 100, '.2f')) + '%')  # show the true cls   format(acc5, '.4f')
        self.lineEdit_13.setText(str(format(time.time() - start, '.2f')) + '秒')

        top_20_img = knn_retrival(self.image_name, model)
        print('top_20_img', top_20_img)
        self.ListWidget.clear()
        for img_path in top_20_img:
            if os.path.isfile(img_path):
                img_name = img_path # .split('/')[-1]
                # cam = QImage(pic[:], pic.shape[1], pic.shape[0], pic.shape[1] * 3, QImage.Format_RGB888)  # 转化为QImage
                # pic = QPixmap(cam).scaled(pic.shape[1], pic.shape[0])  # 设置图片大小
                # self.label_3.setPixmap(pic)
                # self.label_3.setScaledContents(True)
                item = QtWidgets.QListWidgetItem(QIcon(img_path), img_name)  # icon对象
                self.ListWidget.addItem(item)
        try:
            self.ListWidget.setCurrentRow(0)
        except Exception as inst:
            print(inst)

        # if pred_class == self.image_name.split('/')[-2].split('.')[-1]:   # 输出真实类别
        #     print("预测正确，该物种真实类别为{}".format(self.image_name.split('/')[-2].split('.')[-1]))
        # else:
        #     self.lineEdit_13.setText("NO")
        # try:
        #     self.listWidget.setCurrentRow(0)
        # except Exception as inst:
        #     print(inst)

    def dataset_reference(self):
        QtWidgets.QApplication.processEvents()
        self.textBrowser.clear()
        # 演示代码
        if self.radioButton.isChecked() == True:
            data_path = "G:/github_files/datasets/CUB/test"
            dataset_name = "CUB"
        elif self.radioButton_2.isChecked() == True:
            data_path = "G:/github_files/datasets/CAR/test"
            dataset_name = "CAR"
        elif self.radioButton_3.isChecked() == True:
            data_path = "G:/github_files/datasets/Aircraft/test"
            dataset_name = "Aircraft"

        acc1, acc5, times = reference(data_path, dataset_name, self.textBrowser, self.lineEdit_8, func=self.printf)

        self.lineEdit_9.setText(str(acc1) + "%")
        self.lineEdit_10.setText(str(acc5) + "%")
        self.lineEdit_11.setText(str(times) + "秒")


# class Thread(QThread):
#     sinout = pyqtSignal(str)
#     def __init__(self):
#         super(Thread, self).__init__()
#         self.working = True
#
#     def run(self):
#         if self.working == True:
#             = dataset_reference()
#             sleep(1)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 实例化一个应用对象
    window = mywindow()

    image = QtGui.QPixmap()
    image.load(r"./background/star2.png")
    palette1 = QtGui.QPalette()
    palette1.setBrush(window.backgroundRole(), QtGui.QBrush(image)) #背景图片
    # palette1.setColor(window.backgroundRole(), QColor(192, 253, 123))  # 背景颜色
    window.setPalette(palette1)
    window.setAutoFillBackground(True)


    window.setWindowTitle("自监督细粒度图像识别系统")
    window.show()
    sys.exit(app.exec_())   # 确保主循环安全退出