from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import random
import cv2
from PIL import Image
import matplotlib.image as mpimg
import gdal


# from data_argumentation import *
def label_binary(labelimg):
    labelimg /= 255
    labelimg[labelimg > 0.3] = 1
    labelimg[labelimg <= 0.3] = 0
    return labelimg

# Prediction_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Train\\"
# masked_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Val\\"
#
# imgs = glob.glob(masked_DIR + "masked\\*.tif")
# random.shuffle(imgs)
# cnt = 0
# while 1:
#     for imgname in imgs:
#         midname = imgname[imgname.rindex("\\") + 1:]
#         masked = cv2.imread(masked_DIR + "masked\\" + midname)
#         prediction = cv2.imread(Prediction_DIR + "prediction\\" + midname)
#
#         masked = img_to_array(masked).astype('float32')
#         prediction = img_to_array(prediction).astype('float32')
#
#         cha = masked - prediction
#
#         label = img_to_array(label).astype('float32')
#         img /= 255
#         label = label_binary(label)
#         imgdatas.append(img)
#         imglabels.append(label)
#         cnt += 1



def generatedata_0riginal(path,batchsize):
    imgs = glob.glob(path + "masked\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    # num = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            # img = cv2.imread(path + "masked\\" + midname)
            img = mpimg.imread(path + "masked\\" + midname)
            # label = cv2.imread(path + "single-mask\\" + midname, cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(path + "overall-mask\\" + midname,  cv2.IMREAD_COLOR)
            label = mpimg.imread(path + "overall-mask\\" + midname)
            # label = cv2.imread(path + "mask\\118032\\" + midname)
            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img /= 255
            # label = label_binary(label)
            label /= 255
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []
                # num += 2
                # print(num)


def generatedata_0riginal_1(path,batchsize):
    imgs = glob.glob(path + "masked\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = cv2.imread(path + "masked\\" + midname)
            label = cv2.imread(path + "overall-mask\\" + midname, cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(path + "mask\\118032\\" + midname)
            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img /= 255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []

def generatedata(path, batchsize):
    imgs = glob.glob(path + "image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = cv2.imread(path + "image\\" + midname)
            label = cv2.imread(path + "label\\" + midname, cv2.IMREAD_GRAYSCALE)
            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img /= 255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, [labeldatas, labeldatas, labeldatas, labeldatas, labeldatas])
                cnt = 0
                imgdatas = []
                imglabels = []


def generatedata4(path, batchsize):
    imgs = glob.glob(path + "masked\\\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    imglabels_sub2 = []
    imglabels_sub4 = []
    imglabels_sub8 = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = cv2.imread(path + "masked\\" + midname)
            label = cv2.imread(path + "mask\\" + midname, cv2.IMREAD_GRAYSCALE)
            label_sub2 = cv2.resize(label, (256, 256), interpolation=cv2.INTER_AREA)
            label_sub4 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
            label_sub8 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            label_sub2 = img_to_array(label_sub2).astype('float32')
            label_sub4 = img_to_array(label_sub4).astype('float32')
            label_sub8 = img_to_array(label_sub8).astype('float32')
            img /= 255
            label = label_binary(label)
            label_sub2 = label_binary(label_sub2)
            label_sub4 = label_binary(label_sub4)
            label_sub8 = label_binary(label_sub8)
            imgdatas.append(img)
            imglabels.append(label)
            imglabels_sub2.append(label_sub2)
            imglabels_sub4.append(label_sub4)
            imglabels_sub8.append(label_sub8)
            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), [np.array(imglabels_sub8), np.array(imglabels_sub4), np.array(imglabels_sub2),
                                            np.array(imglabels), np.array(imglabels)])
                cnt = 0
                imgdatas = []
                imglabels = []
                imglabels_sub2 = []
                imglabels_sub4 = []
                imglabels_sub8 = []

def generatedata_multichannel(path, batchsize):
    imgs = glob.glob(path + "Image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            #img = cv2.imread(path + "Image\\" + midname)
            label = cv2.imread(path + "Label\\" + midname, cv2.IMREAD_GRAYSCALE)
            label = img_to_array(label).astype('float32')
            label = label_binary(label)
            imglabels.append(label)

            img_dataset = gdal.Open(path + "Image\\" + midname)
            im_width = img_dataset.RasterXSize
            im_height = img_dataset.RasterYSize
            im_data = img_dataset.ReadAsArray(0, 0, im_width, im_height)
            #print(im_data.shape)
            img = np.ndarray((im_width, im_height, 4), dtype=np.float32)
            # img, label = data_argmentation(img, label)
            img[:, :, 0] = im_data[0, :, :]
            img[:, :, 1] = im_data[1, :, :]
            img[:, :, 2] = im_data[2, :, :]
            img[:, :, 3] = im_data[3, :, :]
            #img = img_to_array(img).astype('float32')
            #mg /= (255*255)
            imgdatas.append(img)

            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), np.array(imglabels))
                cnt = 0
                imgdatas = []
                imglabels = []


def generatedata4_multichannel(path, batchsize):
    imgs = glob.glob(path + "Image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    imglabels_sub2 = []
    imglabels_sub4 = []
    imglabels_sub8 = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            #img = cv2.imread(path + "Image\\" + midname)
            img_dataset = gdal.Open(path + "Image\\" + midname)
            im_width = img_dataset.RasterXSize
            im_height = img_dataset.RasterYSize
            im_data = img_dataset.ReadAsArray(0, 0, im_width, im_height)
            # print(im_data.shape)
            img = np.ndarray((im_width, im_height, 4), dtype=np.float32)
            # img, label = data_argmentation(img, label)
            img[:, :, 0] = im_data[0, :, :]
            img[:, :, 1] = im_data[1, :, :]
            img[:, :, 2] = im_data[2, :, :]
            img[:, :, 3] = im_data[3, :, :]

            label = cv2.imread(path + "Label-cloud\\" + midname, cv2.IMREAD_GRAYSCALE)
            label_sub2 = cv2.resize(label, (256, 256), interpolation=cv2.INTER_AREA)
            label_sub4 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
            label_sub8 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
            #img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            label_sub2 = img_to_array(label_sub2).astype('float32')
            label_sub4 = img_to_array(label_sub4).astype('float32')
            label_sub8 = img_to_array(label_sub8).astype('float32')
            img /= (255*255)
            label = label_binary(label)
            label_sub2 = label_binary(label_sub2)
            label_sub4 = label_binary(label_sub4)
            label_sub8 = label_binary(label_sub8)
            imgdatas.append(img)
            imglabels.append(label)
            imglabels_sub2.append(label_sub2)
            imglabels_sub4.append(label_sub4)
            imglabels_sub8.append(label_sub8)
            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), [np.array(imglabels_sub8), np.array(imglabels_sub4), np.array(imglabels_sub2),
                                            np.array(imglabels), np.array(imglabels)])
                cnt = 0
                imgdatas = []
                imglabels = []
                imglabels_sub2 = []
                imglabels_sub4 = []
                imglabels_sub8 = []

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    path = "F:/HED-BSDS"
    f = open("F:/HED-BSDS/train_pair.txt", "r")
    files = f.readlines()
    name = files[0].split(" ")
    img = cv2.imread(os.path.join(path, name[0]))
    label = cv2.imread(os.path.join(path, name[1][:-1]))
    img = cv2.resize(img, (512, 512))
    label = cv2.resize(label, (512, 512))
    cv2.imshow("1", img)
    cv2.imshow("2", label)
    cv2.waitKey()