import gc
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from model import myUnet
# from data import generatedata_0riginal,generatedata_0riginal_1

from model import myUnet
from data import generatedata_0riginal,generatedata_0riginal_1
import numpy as np
import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img

BATCH_SIZE = 1

Train_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Train\\"
Val_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Val\\"
Test_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Test\\"

# Train_DIR = "drive\\MSCN-project\\data-for-detection\\lansat-fintune\\Train\\"
# Val_DIR = "drive\\MSCN-project\\data-for-detection\\lansat-fintune\\Val\\"
# Test_DIR = "drive\\MSCN-project\\data-for-detection\\lansat-fintune\\Test\\"

train_generator = generatedata_0riginal(Train_DIR, BATCH_SIZE)
val_generator = generatedata_0riginal(Val_DIR, BATCH_SIZE)
test_generator = generatedata_0riginal(Test_DIR, BATCH_SIZE)

test = next(test_generator)
(masked, label) = test

(b, h, w, c) = label.shape
label_show = np.zeros([b, h, w, 3*c])

for i in range(len(masked)):
    label_show[i, :, :, :] = np.dstack((label[i, :, :, :], label[i, :, :, :], label[i, :, :, :]))

def plot_callback(model):

    # Get samples & Display them
    mask = model.predict(masked)
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    mask_show = np.zeros([b, h, w, 3*c])
    for i in range(len(masked)):
        # mask[i, :, :, :][mask[i, :, :, :]> 0.8] = 1
        # mask[i, :, :, :][mask[i, :, :, :] <= 0.8] = 0
        mask_array = mask[i, :, :, :]
        mask_array[mask_array >= 0.5] = 1
        mask_array[mask_array < 0.5] = 0
        mask_img = array_to_img(mask[i, :, :, :])
        mask_img.save(r'E:\project\MSCN\log\testsample\result_{}_{}.jpg'.format(i, pred_time))
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i, :, :, :])
        mask_show[i, :, :, :] = np.dstack((mask[i, :, :, :], mask[i, :, :, :], mask[i, :, :, :]))
        # axes[1].imshow(mask[i, :, :, :])
        # axes[2].imshow(label[i, :, :, :])
        axes[1].imshow(mask_show[i, :, :, :] * 255)
        axes[2].imshow(label_show[i, :, :, :] * 255)

        axes[0].set_title('Masked Image')
        axes[1].set_title('mask Image')
        axes[2].set_title('mask_label Image')

        plt.savefig(r'E:\project\MSCN\log\testsample\img_{}_{}.png'.format(i, pred_time))
        plt.close()

model = myUnet(weight_filepath='E:/project/MSCN/log/weight/')

# model.load(r"E:\project\UNet-detection\log\weight\Landsat\346_weights_2018-10-05-21-44-20.h5")

img_num = len(glob.glob(Train_DIR + "masked\\*.tif"))
val_num = len(glob.glob(Val_DIR + "masked\\*.tif"))

model.fit(
    train_generator,
    steps_per_epoch=img_num // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_num //BATCH_SIZE,
    epochs=5000,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='E:/project/MSCN/log/log', write_graph=False)
    ]
)

