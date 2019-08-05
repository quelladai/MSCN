import gc
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import myUnet
from data import generatedata_0riginal,generatedata_0riginal_1
import numpy as np
import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras
import os


BATCH_SIZE = 1

Train_DIR = "E:\\Experiment data\\data\\Landsat\\Train\\"
Val_DIR = "E:\\Experiment data\\data\\Landsat\\Val\\"
Test_DIR = "E:\\Experiment data\\data\\Landsat\\Test\\"

train_generator = generatedata_0riginal(Train_DIR, BATCH_SIZE)
val_generator = generatedata_0riginal(Val_DIR, BATCH_SIZE)
test_generator = generatedata_0riginal(Test_DIR, BATCH_SIZE)

test = next(test_generator)
(masked, label) = test
# print(masked.shape, label.shape)
#
# train_masked_DIR = r"E:\Experiment data\data-for-detection\lansat-fintune\Train\masked\\"
# train_label_DIR = r"E:\Experiment data\data-for-detection\lansat-fintune\Train\single-mask\\"
#
# val_masked_DIR = r"E:\Experiment data\data-for-detection\lansat-fintune\Val\masked\\"
# val_label_DIR = r"E:\Experiment data\data-for-detection\lansat-fintune\Val\single-mask\\"
#
# test_masked_DIR = r"E:\Experiment data\data-for-detection\lansat-fintune\Test\masked\\"
# test_label_DIR = r"E:\Experiment data\data-for-detection\lansat-fintune\Test\single-mask\\"
#
# class DataGenerator_train(ImageDataGenerator):
#     def flow_from_directory(self, directory, *args, **kwargs):
#         generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
#         while True:
#             masked = next(generator)
#
#             data_gen_args = dict(rotation_range=20,
#                                  width_shift_range=0.2,
#                                  height_shift_range=0.2,
#                                  rescale=1. / 255,
#                                  horizontal_flip=True)
#             datagen = ImageDataGenerator(**data_gen_args)
#             mask_generator = datagen.flow_from_directory(
#                 train_label_DIR,
#                 batch_size=BATCH_SIZE,
#                 target_size=(512, 512),
#                 class_mode=None,
#                 shuffle=True,
#                 seed=1)
#
#             mask = next(mask_generator)
#
#             gc.collect()
#             yield masked, mask
#
# train_datagen = DataGenerator_train(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1. / 255,
#     horizontal_flip=True
# )
# train_generator = train_datagen.flow_from_directory(train_masked_DIR ,
#                                                       target_size=(512, 512),
#                                                       batch_size=BATCH_SIZE,
#                                                       shuffle=True,
#                                                       seed=1
#                                                       )
#
#
# class DataGenerator_val(ImageDataGenerator):
#     def flow_from_directory(self, directory, *args, **kwargs):
#         generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
#         while True:
#             masked = next(generator)
#
#             data_gen_args = dict(rotation_range=20,
#                                  width_shift_range=0.2,
#                                  height_shift_range=0.2,
#                                  rescale=1. / 255,
#                                  horizontal_flip=True)
#             datagen = ImageDataGenerator(**data_gen_args)
#             mask_generator = datagen.flow_from_directory(
#                 val_label_DIR,
#                 batch_size=BATCH_SIZE,
#                 target_size=(512, 512),
#                 class_mode=None,
#                 shuffle=True,
#                 seed=1)
#
#             mask = next(mask_generator)
#
#             gc.collect()
#             yield masked, mask
#
#
# val_datagen = DataGenerator_val(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1. / 255,
#     horizontal_flip=True
# )
# val_generator = val_datagen.flow_from_directory(val_masked_DIR,
#                                                 target_size=(512, 512),
#                                                 batch_size=BATCH_SIZE,
#                                                 shuffle=True,
#                                                 seed=1
#                                                 )
#
#
# class DataGenerator_test(ImageDataGenerator):
#     def flow_from_directory(self, directory, *args, **kwargs):
#         generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
#         while True:
#             masked = next(generator)
#
#             data_gen_args = dict(rescale=1. / 255)
#             datagen = ImageDataGenerator(**data_gen_args)
#             mask_generator = datagen.flow_from_directory(
#                 test_label_DIR,
#                 batch_size=BATCH_SIZE,
#                 target_size=(512, 512),
#                 class_mode=None,
#                 shuffle=True,
#                 seed=1)
#
#             mask = next(mask_generator)
#
#             gc.collect()
#             yield masked, mask
#
#
# test_datagen = DataGenerator_test(rescale=1. / 255)
# test_generator = test_datagen.flow_from_directory(test_masked_DIR,
#                                                   target_size=(512, 512),
#                                                   batch_size=BATCH_SIZE,
#                                                   shuffle=True,
#                                                   seed=1
#                                                   )
#
#
# test= next(test_generator)
# (masked, label) = test
# print(label.shape)
#
# (b, h, w, c) = label.shape
# label_show = np.zeros([b, h, w, 3*c])

# for i in range(len(masked)):
#     label_show[i, :, :, :] = np.dstack((label[i, :, :, :], label[i, :, :, :], label[i, :, :, :]))

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            #  val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.show()

def threshold_img(img):
    x = np.zeros((patch_size, patch_size, 3))
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            if (img[i][j][0] >= 0.001) and (img[i][j][0]>=img[i][j][1]):
                x[i][j][0] = 1
                x[i][j][1] = 0
                x[i][j][2] = 0
            elif (img[i][j][1] >= 0.001) and (img[i][j][0]<img[i][j][1]):
                x[i][j][0] = 0
                x[i][j][1] = 1
                x[i][j][2] = 0
            else:
                x[i][j][0] = 0
                x[i][j][1] = 0
                x[i][j][2] = 1
    return x

def max_img(img):
    x = np.zeros((patch_size, patch_size, 3))
    # cloud = np.ndarray((patch_size, patch_size))
    # shadow = np.ndarray((patch_size, patch_size))
    # background = np.ndarray((patch_size, patch_size))
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            a = img[i][j]
            b = np.argmax(a)
            x[i][j][b] = 1
    return x

patch_size = 512

def plot_callback(model):

    # Get samples & Display them
    mask = model.predict(masked)
    (b, h, w, c) = mask.shape
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    label_normal = np.ndarray((b, h, w, c), dtype=np.uint8)

    # Clear current output and display test images
    # mask_show = np.zeros([b, h, w, 3*c])
    for i in range(len(masked)):
        labelimg = mask[i]
        label_normal[i] = max_img(labelimg)
        # print(label_normal.shape)
        # mask[i, :, :, :][mask[i, :, :, :]> 0.8] = 1
        # mask[i, :, :, :][mask[i, :, :, :] <= 0.8] = 0
        # mask_array = mask[i, :, :, :]
        # mask_array[mask_array >= 0.5] = 1
        # mask_array[mask_array < 0.5] = 0
        # mask_img = array_to_img(mask[i, :, :, :])
        # mask_img.save(r'E:\project\MSCN\log\testsample\result_{}_{}.jpg'.format(i, pred_time))
        # mask_show = np.zeros([b, h, w, c])
        # print(label_normal.shape)
        # mask_show[i, :, :, :] = np.dstack((label_normal[i, :, :], label_normal[i, :, :], label_normal[i, :, :]))
        # mask_img = array_to_img(mask_show[i, :, :, :] * 255)
        # mask_img.save(r'E:\project\UNet-detection\log\classification\testsample\result_{}_{}.jpg'.format(i, pred_time))
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i, :, :, :])
        # mask_show[i, :, :, :] = np.dstack((mask[i, :, :, :], mask[i, :, :, :], mask[i, :, :, :]))
        axes[1].imshow(label_normal[i, :, :, :] * 255)
        axes[2].imshow(label[i, :, :, :] * 255)
        # axes[1].imshow(mask_show[i, :, :, :] * 255)
        # axes[2].imshow(label_show[i, :, :, :] * 255)

        axes[0].set_title('Masked Image')
        axes[1].set_title('mask Image')
        axes[2].set_title('mask_label Image')

        plt.savefig(r'G:\Detection\Net1\MSCN\img_{}_{}.png'.format(i, pred_time))
        plt.close()


model = myUnet(weight_filepath
               ='G:/Detection/Net1/MSCN/weight/')
# model.summary()

model.load(r"G:\Detection\Net1\MSCN\weight\173_weights_2018-11-15-16-26-40.h5")

img_num = len(glob.glob(Train_DIR + "masked\\*.tif"))
val_num = len(glob.glob(Val_DIR + "masked\\*.tif"))

history = LossHistory()

checkpoint_fn = os.path.join(
    'G:\\Detection\\Net1\\MSCN\\bestweight\\checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.h5')
checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_acc', mode='auto', save_best_only='True')
tensorboard = TensorBoard(log_dir='G:/Detection/Net1/MSCN/log', write_graph=False)
callbacks_list = [checkpoint, tensorboard, history]
# callbacks_list = [checkpoint, tensorboard]

model.fit(
    train_generator,
    steps_per_epoch=img_num// BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_num //BATCH_SIZE,
    epochs=827,
    plot_callback=plot_callback,
    callbacks=callbacks_list
)

history.loss_plot('epoch')