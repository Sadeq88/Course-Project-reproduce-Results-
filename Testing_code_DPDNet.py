import numpy as np
import os
import math
import cv2
import random
from skimage import measure
from sklearn.metrics import mean_absolute_error
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend
from keras.layers import Add




# Paths
# READ & WRITE DATA PATHS
subfolders=['source/','target/']
# path to save model
path_save_model='./ModelCheckpoints/defocus_deblurring_dp_retrained.hdf5'
# paths to read data
train_data_path = './DPDNet_dataset_org/'
validation_data_path = './DPDNet_dataset_org/'
# path to write results
path_write='./results/retrained_DPDNet/'


# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS	    #
total_nb_test = len([validation_data_path + 'test_c/' + subfolders[0] + f for f
                    in os.listdir(validation_data_path + 'test_c/' + subfolders[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])

# input image size
img_width, img_height = 1680, 1120
# mean value pre-claculated
source_mean = 0
# number of input and output channels
num_channels_all, num_channels = 6, 3 # change conv9 in the model and the folowing variable
# "-1" to read image as it including bit depth
color_flag=-1
bit_depth, normalization_value = 16, (2 ** 16) - 1

# Data sets and evaluation results
train_set, val_set, test_set = [], [], []
size_set, portrait_orientation_set = [], []
mse_values, psnr_values, ssim_values, mae_values = [], [], [], []


# # Evaluation functions:MAE, MSE_PSNR_SSIM, 
def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def MSE_PSNR_SSIM(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    PIXEL_MAX = 1
    return mse_, 10 * math.log10(PIXEL_MAX / mse_), measure.compare_ssim(img1,
                                                    img2, data_range=PIXEL_MAX,
                                                    multichannel=True)


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

            
            
def test_generator(num_image):
    in_img_tst = np.zeros((num_image, img_height, img_width, num_channels_all))
    out_img_gt = np.zeros((num_image, img_height, img_width, num_channels))
    resize_flag=True
    for i in range(num_image):
        print('Read image: ',i,num_image)
        if resize_flag:
            temp_img_l=cv2.imread(test_set[i][1],color_flag)
            size_set.append([temp_img_l.shape[1],temp_img_l.shape[0]])
            if temp_img_l.shape[0]>temp_img_l.shape[1]:
                portrait_orientation_set.append(True)
                temp_img_l=cv2.rotate(temp_img_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
                in_img_tst[i, :,:,0:3] = (cv2.resize((temp_img_l-source_mean)/normalization_value,
                            (img_width,img_height))).reshape((img_height, img_width,num_channels))
                temp_img_r=cv2.rotate(cv2.imread(test_set[i][2],color_flag), cv2.ROTATE_90_COUNTERCLOCKWISE)
                in_img_tst[i, :,:,3:6] = (cv2.resize((temp_img_r-source_mean)
                                    /normalization_value,(img_width,img_height))).reshape((img_height, img_width,num_channels))
                temp_img_trg=cv2.rotate(cv2.imread(test_set[i][3],color_flag), cv2.ROTATE_90_COUNTERCLOCKWISE)
                out_img_gt[i, :] = (cv2.resize((temp_img_trg-source_mean)
                                    /normalization_value,(img_width,img_height))).reshape((img_height, img_width,num_channels))

            else:
                portrait_orientation_set.append(False)
                in_img_tst[i, :,:,0:3] = (cv2.resize((temp_img_l-source_mean)/normalization_value,
                            (img_width,img_height))).reshape((img_height, img_width,num_channels))
                in_img_tst[i, :,:,3:6] = (cv2.resize((cv2.imread(test_set[i][2],color_flag)-source_mean)
                                    /normalization_value,(img_width,img_height))).reshape((img_height, img_width,num_channels))
                out_img_gt[i, :] = (cv2.resize((cv2.imread(test_set[i][3],color_flag)-source_mean)
                                    /normalization_value,(img_width,img_height))).reshape((img_height, img_width,num_channels))
                
        else:
            in_img_tst[i, :,:,0:3] = ((cv2.imread(test_set[i][1],color_flag)-source_mean)
                                    /normalization_value).reshape((img_height, img_width,num_channels))
            in_img_tst[i, :,:,3:6] = ((cv2.imread(test_set[i][2],color_flag)-source_mean)
                                    /normalization_value).reshape((img_height, img_width,num_channels))
            out_img_gt[i, :] = ((cv2.imread(test_set[i][3],color_flag)-source_mean)
                              /normalization_value).reshape((img_height, img_width,num_channels))
    return in_img_tst, out_img_gt


        
        
def save_eval_predictions(path_to_save,test_imgaes,predictions,gt_images):
    global mse_values, psnr_values, ssim_values, test_set
    for i in range(len(test_imgaes)):
        mse, psnr, ssim = MSE_PSNR_SSIM((gt_images[i]).astype(np.float64), (predictions[i]).astype(np.float64))
        mae = MAE((gt_images[i]).astype(np.float64), (predictions[i]).astype(np.float64))
        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mae_values.append(mae)

        temp_in_img=cv2.imread(test_set[i][0],color_flag)
        if bit_depth == 8:
            temp_out_img=((predictions[i]*normalization_value)+source_mean).astype(np.uint8)
            temp_gt_img=((gt_images[i]*normalization_value)+source_mean).astype(np.uint8)
        elif bit_depth == 16:
            temp_out_img=((predictions[i]*normalization_value)+source_mean).astype(np.uint16)
            temp_gt_img=((gt_images[i]*normalization_value)+source_mean).astype(np.uint16)
        img_name=((test_set[i][0]).split('/')[-1]).split('.')[0]
        if resize_flag:
            if portrait_orientation_set[i]:
                temp_out_img=cv2.resize(cv2.rotate(temp_out_img,cv2.ROTATE_90_CLOCKWISE),(size_set[i][0],size_set[i][1]))
                temp_gt_img=cv2.resize(cv2.rotate(temp_gt_img,cv2.ROTATE_90_CLOCKWISE),(size_set[i][0],size_set[i][1]))
            else:
                temp_out_img=cv2.resize(temp_out_img,(size_set[i][0],size_set[i][1]))
                temp_gt_img=cv2.resize(temp_gt_img,(size_set[i][0],size_set[i][1]))
        cv2.imwrite(path_to_save+str(img_name)+'_i.png',temp_in_img)
        cv2.imwrite(path_to_save+str(img_name)+'_p.png',temp_out_img)
        cv2.imwrite(path_to_save+str(img_name)+'_g.png',temp_gt_img)
        print('Write image: ',i,len(test_imgaes))
        

# Main execution
shuffle_data('test')

model = load_model(path_save_model, compile=False)

# Fix input layer size
model.layers.pop(0)
input_size = (img_height, img_width, num_channels_all)
input_test = Input(input_size)
output_test = model(input_test)
model = Model(input=input_test, output=output_test)

# image mini-batch size
img_mini_b = 5

test_images, gt_images = test_generator(total_nb_test)
predictions = model.predict(test_images, img_mini_b, verbose=1)

save_eval_predictions(path_write, test_images, predictions, gt_images)

np.save(path_write + 'mse_arr', np.asarray(mse_values))
np.save(path_write + 'psnr_arr', np.asarray(psnr_values))
np.save(path_write + 'ssim_arr', np.asarray(ssim_values))
np.save(path_write + 'mae_arr', np.asarray(mae_values))
np.save(path_write + 'final_eval_arr', [np.mean(np.asarray(mse_values)),
                                        np.mean(np.asarray(psnr_values)),
                                        np.mean(np.asarray(ssim_values)),
                                        np.mean(np.asarray(mae_values))])