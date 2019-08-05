import gc
from copy import deepcopy
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from model import myUnet
import cv2
import glob
import data
import math

Train_DIR = "E:\\Experiment data\\data-for-detection\\Landsat\\Train\\"
Val_DIR = "E:\\Experiment data\\data-for-detection\\Landsat\\Val\\"
Test_DIR = "E:\\Experiment data\\data-for-detection\\Landsat\\Test\\"
Result_DIR = "E:\\Experiment data\\data-for-detection\\Landsat\\Test\\maskMSCN-2sec\\"

model = myUnet()
model.load(r"E:\project\MSCN-1\log\weight-2\1000_weights_2018-10-18-20-48-29.h5")

imgsname = glob.glob(Test_DIR + "masked\\*.tif")
imgdatas = np.ndarray((1, 512, 512, 3), dtype=np.float32)
mask1datas = np.ndarray((1, 512, 512, 3), dtype=np.float32)
temporaldatas = np.ndarray((1, 512, 512, 3), dtype=np.float32)
predictiondatas = np.ndarray((1, 512, 512, 3), dtype=np.float32)

def label_binary(labelimg):
    labelimg /= 255
    labelimg[labelimg > 0.3] = 1
    labelimg[labelimg <= 0.3] = 0
    return labelimg

num = 0
for imgname in imgsname:
    name = imgname[imgname.rindex("\\") + 1:]
    img = cv2.imread(Test_DIR + "masked\\" + name)
    mask1 = cv2.imread(Test_DIR + "maskMSCN\\" + name,  cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(Test_DIR + "prediction\\" + name)
    temporal = cv2.imread(Test_DIR + "temporal\\" + name)

    img = img_to_array(img).astype('float32')
    mask1 = img_to_array(mask1).astype('float32')
    prediction = img_to_array(prediction).astype('float32')
    temporal = img_to_array(temporal).astype('float32')

    imgdatas[0] = img / 255
    predictiondatas[0] = prediction / 255
    temporaldatas[0] = temporal / 255
    mask1datas[0] = label_binary(mask1)
    # mask = model.predict(imgdatas)
    mask = model.predict([imgdatas, temporaldatas, predictiondatas, mask1datas])

    # mask_fin = mask[3][0]
    # mask_fin[mask_fin > 0.3] = 1
    # mask_fin[mask_fin <= 0.3] = 0

    mask = mask[0, :, :, :]
    mask[mask > 0.3] = 1
    mask[mask <= 0.3] = 0

    mask_img = array_to_img(mask)
    mask_img.save(Result_DIR + "\\%s" % name)

    num = num + 1
    if num % 100 == 0:
        print("%d输出完毕！" % num)

    print("输出完毕！")



