from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import glob
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def pixelacc(result, lable):
    data_diff = result - lable
    # data_diff[data_diff == 0] = 1
    # data_diff[data_diff != 1] = 0
    # pix_acc = sum(sum(data_diff))/512/512
    pix_acc = np.sum(data_diff == 0) / 512/512/3
    return pix_acc

def zhankai(img):
    (h, w, c) = img.shape
    imgw = np.ndarray((h, c * w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            for m in range(c):
                imgw[i, j+m*512] = img[i, j, m]
    return imgw

def zhankai2(img):
    (h, w, c) = img.shape
    imgw = np.ndarray((h, (c-1) * w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            for m in range(c-1):
                imgw[i, j+m*512] = img[i, j, m]
    return imgw

def shadow(img):
    (h, w, c) = img.shape
    imgw = np.ndarray((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            imgw[i, j] = img[i, j, 1]
    return imgw

def cloud(img):
    (h, w, c) = img.shape
    imgw = np.ndarray((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            imgw[i, j] = img[i, j, 2]
    return imgw



# 1、计算检测结果的重叠区域并集：两张图像相加，非0的区域
# 2、计算正确检测区域：result图把所有0变为1，相减，值为0的区域

def IOU(result, lable):
    img_sum = result + lable
    # img_sum[img_sum != 0] = 1
    # result[result == 0] = 2
    # dif = result - lable
    # # # Iou_acc = np.sum(dif == 0) / np.sum(img_sum != 0)
    # return [np.sum(dif == 0), np.sum(img_sum != 0)]
    return [np.sum(img_sum >= 510), np.sum(img_sum >= 255)]



# 检测率：正确的房屋像元/模板
# 漏检率：
# 误检率：检测错误的像元/检测总像元
def checkacc(result, lable):
    lable[lable == 255] = 1
    dif = result - lable
    # jc_acc = np.sum(dif == 254) / np.sum(lable == 1)
    # lj_acc = 1 - jc_acc
    # wj_acc = np.sum(dif == 255) / np.sum(result == 255)
    return [np.sum(dif == 254), np.sum(lable == 1), np.sum(dif == 255), np.sum(result == 255)]
    # 检测正确的，总共待检测的个数，检测错误的（多检测的），总共检测出来的个数


def evaluate():
    lable_path = "E:\\Experiment data\\data\\Net3\\Test\\overall-mask\\"
    result_path = "E:\\Experiment data\\data\\Landsat\\Test\\maskMSCN\\"
    imgsname = glob.glob(lable_path + "*.tif")
    totall_pix_acc = 0
    Iou = []
    jc = []
    threshold = 0.3

    n = 0
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        # img = load_img(result_path + name, grayscale=True)
        # img_gt = load_img(imgname, grayscale=True)
        img = load_img(result_path + name)
        img_gt = load_img(imgname)

        imgdata = img_to_array(img).astype(np.float32)
        imgdata_gt = img_to_array(img_gt).astype(np.float32)
        imgdata[imgdata > 255 * threshold] = 255
        imgdata[imgdata <= 255 * threshold] = 0
        imgdata_gt[imgdata_gt > 255 * threshold] = 255
        imgdata_gt[imgdata_gt <= 255 * threshold] = 0
        imgdata_gt = zhankai(imgdata_gt)
        imgdata = zhankai(imgdata)

        # saveimg = array_to_img(imgdata)
        # saveimg.save("E:\\project\\MSCN\\classification-log\\test\\"+name)

        pix_acc = pixelacc(imgdata, imgdata_gt)
        totall_pix_acc += pix_acc
        Iou.append(IOU(imgdata, imgdata_gt))
        jc.append(checkacc(imgdata, imgdata_gt))
        n += 1
        # print(n, "张图像计算完毕！", Iou)
        # if n % 200 == 0:
        #     print(n, "张图像计算完毕！")
    Iou2 = sum(np.array(Iou))
    Iou_acc = Iou2[0]/Iou2[1]
    jc2 = sum(np.array(jc))
    jc_acc = jc2[0]/jc2[1]  # 检测正确的/总共待检测的
    wj_acc = jc2[2]/jc2[3] # 检测错误的/总共检测出来的
    totall_pix_acc = totall_pix_acc / len(imgsname)
    print("Iou 精度: ", Iou_acc)
    print("检测率 recall: ", jc_acc)
    print("误检率", wj_acc)
    print("1-误检率=precision: ", 1 - wj_acc)
    print("像素精度: ", totall_pix_acc)


def evaluate_pj():
    lable_path = "F:\\python_progect\\unet\\resample2\\test\\label\\"
    result_path = "F:\\python_progect\\unet\\results\\"
    imgsname = glob.glob(lable_path + "*.tif")
    totall_pix_acc = 0
    Iou = []
    jc = []
    threshold = 0.55
    n = 0
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(result_path + name, grayscale=True)
        img_gt = load_img(imgname, grayscale=True)
        label_sub2 = cv2.resize(img_gt, (256, 256), interpolation=cv2.INTER_AREA)
        label_pin_sub2 = np.zeros((512, 512), img_gt.dtype)
        label_pin_sub2[0:256, 0:256] = label_sub2
        label_pin_sub2[0:256, 256:512] = label_sub2
        label_pin_sub2[256:512, 0:256] = label_sub2
        label_pin_sub2[256:512, 256:512] = label_sub2

        imgdata = img_to_array(img).astype(np.float32)
        imgdata_gt = img_to_array(label_pin_sub2).astype(np.float32)
        imgdata[imgdata > 255 * threshold] = 255
        imgdata[imgdata <= 255 * threshold] = 0
        pix_acc = pixelacc(imgdata, imgdata_gt)
        totall_pix_acc += pix_acc
        Iou.append(IOU(imgdata, imgdata_gt))
        jc.append(checkacc(imgdata, imgdata_gt))
        n += 1
        if n % 200 == 0:
            print(n, "张图像计算完毕！")
    Iou2 = sum(np.array(Iou))
    Iou_acc = Iou2[0]/Iou2[1]
    jc2 = sum(np.array(jc))
    jc_acc = jc2[0]/jc2[1]
    wj_acc = jc2[2]/jc2[3]
    totall_pix_acc = totall_pix_acc / len(imgsname)
    print("Iou 精度: ", Iou_acc)
    print("检测率 recall: ", jc_acc)
    print("1-误检率=precision: ", 1 - wj_acc)
    print("像素精度: ", totall_pix_acc)


def evaluate_single():
    # lable_path = "E:\\ArcGIS\\resample2_512\\train\\lable\\"
    # lable_path = "E:\\ArcGIS\\building\\2\\label\\label\\"
    lable_path = "E:\\AerialImageDataset\\train\\all\\vienna\\label\\"
    result_path = "E:\\TensorFlow\\unet\\results\\"
    # resultimg_path = "E:\\TensorFlow\\unet\\resultimg\\"
    imgsname = glob.glob(lable_path + "*.tif")
    all_acc = []
    threshold = 0.25
    f = open("single_acc.txt", "w")
    f.write("文件名,Iou,检测率Recall,1-误检率precision,像素精度\n")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(result_path + name, grayscale=True)
        img_gt = load_img(imgname, grayscale=True)
        imgdata = img_to_array(img).astype(np.float32)
        imgdata_gt = img_to_array(img_gt).astype(np.float32)
        imgdata[imgdata > 255 * threshold] = 255
        imgdata[imgdata <= 255 * threshold] = 0

        pix_acc = pixelacc(imgdata, imgdata_gt)
        iou_out = IOU(imgdata, imgdata_gt)
        iou_acc = iou_out[0]/iou_out[1]
        check_out = checkacc(imgdata, imgdata_gt)
        if check_out[1] == 0:
            jc_acc = 0
            wj_acc = 1
        else:
            jc_acc = check_out[0]/check_out[1]
            wj_acc = check_out[2] / check_out[3]
        lj_acc = 1 - jc_acc
        if np.sum(imgdata_gt == 0) != 512*512:
            f.write(name + "," + str(iou_acc) + "," + str(jc_acc) + "," + str(1-wj_acc) + "," + str(pix_acc) + "\n")
        # all_acc.append(acc)
        # acc_np = np.array(all_acc)
    f.close()
    # print(acc_np)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    evaluate()
    # evaluate_single()