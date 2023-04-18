#import statements: importing necessary packages
import os
import math
import random
import numpy as np
import cv2
from skimage import measure
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import Input, Conv2D, ReLU, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler




# Paths
subfolders = ['source/', 'target/']
resize_images = False
#paths
model_checkpoint_path = f'./ModelCheckpoints/defocus_deblurring_dp_retrained.hdf5'
train_data_path = f'./DPDNet_dataset_patch/'
validation_data_path = f'./DPDNet_dataset_patch/'
results_path = f'./results/retrained_DPDNet_patch/'


# input patch size
patch_width, patch_height = 512, 512
# mean value pre-claculated
source_mean, target_mean = 0, 0
# number of epochs
num_epochs = 150
## number of input and output channels
num_channels_all, num_channels = 6, 3
# color flag:"1" for 3-channel 8-bit image or "0" for 1-channel 8-bit grayscale
# or "-1" to read image as it including bit depth
color_flag = -1
normalization_value = (2 ** 16) - 1
# after how many epochs you change learning rate
scheduling_rate, dropout_rate = 60, 0.4

# Generate learning rate array
learning_rates = []
learning_rates.append(2e-5)  # initial learning rate
for i in range(int(num_epochs / scheduling_rate)):
    learning_rates.append(learning_rates[i] * 0.5)

train_set, val_set = [], []

batch_size = 1
total_train_images = len([f for f in os.listdir(train_data_path + 'train_c/' + subfolders[0])
                          if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.TIF'))])
total_val_images = len([f for f in os.listdir(validation_data_path + 'val_c/' + subfolders[0])
                        if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.TIF'))])
num_train_batches = int(math.ceil(total_train_images / batch_size))
num_val_batches = int(math.ceil(total_val_images / batch_size))



def unet(input_data):
    def convolution_block(filters, input_layer):
        conv_layer = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(input_layer)
        conv_layer = ReLU()(conv_layer)
        conv_layer = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(conv_layer)
        return ReLU()(conv_layer)

    block1 = convolution_block(64, input_data)
    pooling1 = MaxPooling2D(pool_size=(2, 2))(block1)

    block2 = convolution_block(128, pooling1)
    pooling2 = MaxPooling2D(pool_size=(2, 2))(block2)

    block3 = convolution_block(256, pooling2)
    pooling3 = MaxPooling2D(pool_size=(2, 2))(block3)

    block4 = convolution_block(512, pooling3)
    dropout4 = Dropout(dropout_rate)(block4)
    pooling4 = MaxPooling2D(pool_size=(2, 2))(dropout4)

    block5 = convolution_block(1024, pooling4)
    dropout5 = Dropout(dropout_rate)(block5)

    upsample6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(dropout5))
    merge6 = concatenate([dropout4, upsample6], axis=3)

    block6 = convolution_block(512, merge6)

    upsample7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(block6))
    merge7 = concatenate([block3, upsample7], axis=3)

    block7 = convolution_block(256, merge7)

    upsample8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(block7))
    merge8 = concatenate([block2, upsample8], axis=3)

    block8 = convolution_block(128, merge8)

    upsample9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(block8))
    merge9 = concatenate([block1, upsample9], axis=3)

    block9 = convolution_block(64, merge9)
    block9 = Conv2D(3, 3, padding='same', kernel_initializer='he_normal')(block9)
    block9 = ReLU()(block9)

    output_layer = Conv2D(num_channels, 1, activation='sigmoid')(block9)

    return output_layer


# functions for data preparation, training, and validation of the model.
def check_dir(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def schedule_learning_rate(epoch):
    lr = learning_rates[int(epoch / scheduling_rate)]
    return lr


def shuffle_data(data_kind):
    global train_set, val_set, test_set, train_data_path, validation_data_path
    if data_kind == 'train':
        path_read = train_data_path
    else:
        path_read = validation_data_path

    center_src_images = [path_read + data_kind + '_c/' + subfolders[0] + f for f
                         in os.listdir(path_read + data_kind + '_c/' + subfolders[0])
                         if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.TIF'))]
    center_src_images.sort()

    center_trg_images = [path_read + data_kind + '_c/' + subfolders[1] + f for f
                         in os.listdir(path_read + data_kind + '_c/' + subfolders[1])
                         if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.TIF'))]
    center_trg_images.sort()

    left_src_images = [path_read + data_kind + '_l/' + subfolders[0] + f for f
                       in os.listdir(path_read + data_kind + '_l/' + subfolders[0])
                       if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.TIF'))]
    left_src_images.sort()

    right_src_images = [path_read + data_kind + '_r/' + subfolders[0] + f for f
                        in os.listdir(path_read + data_kind + '_r/' + subfolders[0])
                        if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.TIF'))]
    right_src_images.sort()

    num_images = len(center_src_images)

    # Generate random shuffle index list for all lists
    random_indices = np.arange(num_images)
    random.shuffle(random_indices)

    center_src_images = np.asarray(center_src_images)[random_indices]
    center_trg_images = np.asarray(center_trg_images)[random_indices]

    left_src_images = np.asarray(left_src_images)[random_indices]
    right_src_images = np.asarray(right_src_images)[random_indices]

    for i in range(num_images):
        if data_kind == 'train':
            train_set.append([center_src_images[i], left_src_images[i], right_src_images[i],
                              center_trg_images[i]])
        elif data_kind == 'val':
            val_set.append([center_src_images[i], left_src_images[i], right_src_images[i],
                            center_trg_images[i]])
        elif data_kind == 'test':
            test_set.append([center_src_images[i], left_src_images[i], right_src_images[i],
                             center_trg_images[i]])
        else:
            raise NotImplementedError


def data_generator(phase='train'):
    if phase == 'train':
        dataset_temp = train_set
        num_images = total_train_images
    elif phase == 'val':
        dataset_temp = val_set
        num_images = total_val_images
    else:
        raise NotImplementedError

    img_counter = 0
    input_images = np.zeros((batch_size, patch_height, patch_width, num_channels_all))
    target_images = np.zeros((batch_size, patch_height, patch_width, num_channels))
    
    while True:
        for i in range(0, batch_size):
            source_img_center = dataset_temp[(img_counter + i) % (num_images)][0]
            source_img_left = dataset_temp[(img_counter + i) % (num_images)][1]
            source_img_right = dataset_temp[(img_counter + i) % (num_images)][2]

            target_img = dataset_temp[(img_counter + i) % (num_images)][3]
            
            if resize_images:
                input_images[i, :, :, 0:3] = (cv2.resize((cv2.imread(source_img_left, color_flag) - source_mean)
                                          / normalization_value, (patch_width, patch_height))).reshape((patch_height, patch_width, num_channels))
                input_images[i, :, :, 3:6] = (cv2.resize((cv2.imread(source_img_right, color_flag) - source_mean)
                                          / normalization_value, (patch_width, patch_height))).reshape((patch_height, patch_width, num_channels))
                target_images[i, :] = (cv2.resize((cv2.imread(target_img, color_flag) - target_mean)
                                          / normalization_value, (patch_width, patch_height))).reshape((patch_height, patch_width, num_channels))
            else:
                input_images[i, :, :, 0:3] = ((cv2.imread(source_img_left, color_flag) - source_mean)
                                            / normalization_value).reshape((patch_height, patch_width, num_channels))
                input_images[i, :, :, 3:6] = ((cv2.imread(source_img_right, color_flag) - source_mean)
                                            / normalization_value).reshape((patch_height, patch_width, num_channels))
                target_images[i, :] = ((cv2.imread(target_img, color_flag) - target_mean)
                                      / normalization_value).reshape((patch_height, patch_width, num_channels))

        yield (input_images, target_images)
        img_counter = (img_counter + batch_size) % (num_images)


def train():
    check_dir(results_path)

    shuffle_data('train')
    shuffle_data('val')

    input_data = Input(batch_shape=(None, patch_height, patch_width, num_channels_all))
    model = Model(inputs=input_data, outputs=unet(input_data))
    model.summary()
    model.compile(optimizer=Adam(lr=learning_rates[0]), loss='mean_squared_error')

    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
    learning_rate_scheduler_callback = LearningRateScheduler(schedule=schedule_learning_rate)

    history = model.fit_generator(data_generator('train'), num_train_batches, num_epochs,
                                  validation_data=data_generator('val'),
                                  validation_steps=num_val_batches, callbacks=[model_checkpoint, learning_rate_scheduler_callback])

    np.save(output_path + 'train_loss_arr', history.history['loss'])
    np.save(output_path + 'val_loss_arr', history.history['val_loss'])

if __name__ == "__main__":
    train()


