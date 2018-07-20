# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg  # mpimg
import cv2
import os
from PIL import Image


def addLogo():
    img1 = cv2.imread('.\pic.jpg')  # 原始图像
    img2 = cv2.imread('.\LOGO.png', -1)  # logo图像，要往原始图像上添加
    img3 = cv2.imread('.\LOGO.png')

    image_s4 = cv2.split(img2)[3]  # BGRA-----A
    rows, cols, channels = img2.shape  # 得到logo的尺寸
    roi = img1[0:rows, 0:cols].copy()  # 在原始图像中截取logo图像大小的部分
    img2_RGB = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)  # 将logo图像灰度化
    #img2_HSV = cv2.cvtColor(img2_RGB, cv2.COLOR_BGR2HSV)

    # 将logo灰度图二值化，将得到的图像赋值给mask，logo部分的值为255，白色
    ret, mask = cv2.threshold(image_s4, 254, 255, cv2.THRESH_BINARY)
    ret, thresh3 = cv2.threshold(image_s4, 235, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image_s4, 35, 255, cv2.THRESH_TOZERO)
    '''
    titles = ['img','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [image_s4,mask,thresh2,thresh3,thresh4,thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    '''
    image_mer1 = cv2.merge((thresh4, thresh4, thresh4))

    #cv2.imwrite("faces/" + "LOGO-mer.jpg", image_mer1)
    #mask_2 = cv2.adaptiveThreshold(image_s4,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,1)
    mask_inv = cv2.bitwise_not(thresh4)  # 将mask按位取反，即白变黑，黑变白

    #kernel = np.ones((3,3),np.uint8)
    #img2_RGB = cv2.erode(img2_RGB,kernel)#腐蚀处理

    # 将原始图像中截取的部分做处理，mask_inv中黑色部分按位与运算，即保留黑色部分，保留除logo位置外的部分
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # 将logo图像中，thresh4部分按位与运算，即保留黑色部分，保留logo
    img2_fg = cv2.bitwise_and(img3, img3, mask=thresh4)
    dst = cv2.add(img1_bg, img2_fg)  # 图像相加
    img1[0:rows, 0:cols, :] = dst  # 图像替换
    # 异或运算，保留LOGO图的边缘部分，没有该操作会损失LOGO图边缘颜色较浅的部分
    pic_xor = cv2.bitwise_xor(img1_bg, image_mer1, mask=mask_inv)
    dst1 = cv2.add(pic_xor, img2_fg)
    #mpimg.imsave("faces/" + "LOGO-1-2-xor.png", pic_xor1,dpi = 100)
    img1[0:rows, 0:cols, :] = dst1
    #cv2.imwrite("faces/" + "LOGO-1-2.jpg", img1) #jpg格式图片有像素损失，边缘有颜色较深像素
    dst2 = cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB)
    img1[0:rows, 0:cols, :] = dst2
    plt.imshow(img1)
    plt.show()
    mpimg.imsave("faces/" + "LOGO-1-2-mp.png",
                 img1, dpi=100)  # png格式不会有像素损失及黑色边框
