# Load packages:
import torch
from tensorflow import keras
from skimage.util import random_noise
import pandas as pd
import os
from monai.utils.enums import NumpyPadMode
from monai.transforms import(AsChannelFirstd, CenterSpatialCropd, Compose, Resized, ScaleIntensityd, ToTensord,
                             Rotated, RandZoomd, RandAdjustContrastd, Identityd, GaussianSmoothd, RandZoomd, LoadImaged)
import torchvision
import numpy as np

# Convert labels from 1 channel with 5 class values to 5 channels with binary values (one-hot encoded)
def ToCategoricald(data_dict, n_classes=5):

        #round interpolated data to nearest class
        data_dict["label"] = torch.round(data_dict["label"])

        #flip to channel last:
        trans_label = torch.transpose(data_dict['label'],0,-1)

        #add tensor dimension
        squeeze_label = torch.squeeze(trans_label)
        
        #transform to one-hot encoded
        encoded_label = torch.from_numpy(keras.utils.to_categorical(squeeze_label.numpy(), num_classes=n_classes))

        #return categorical label in dictionary
        data_dict["label"] = torch.transpose(encoded_label,0,-1) #flip back to channel first

        return data_dict


# Custom function that adds noise to image (contained within a dictionary):
def add_noise(dict):
    #Load image as numpy array:
    img = dict["image"]
    if torch.is_tensor(img):
        numpy_img = img.numpy()
    elif img.dtype == 'float32':
        numpy_img = img
    else:
        print('Warning: Image datatype {} is not expected, Image should be float32 or tensor'.format(img.dtype))

    #Add noise to numpy array:
    noised_numpy_img = random_noise(numpy_img) #expects an ndarray
    
    #Create new dictionary, fill with noisy image and original label
    new_dict = {} 
    new_dict["image"] = torch.from_numpy(noised_numpy_img).float()
    for key in dict.keys():
        if key != 'image':
            new_dict[key] = dict[key]

    return new_dict


# Custom function that converts image (contained within a dictionary) to sepia tones:
def sepia(dict):
    #This is designed to act on a single image, and applied in the same way as a monai transform
    
    #Initialize empty tensor of sepia pixels, get original RGB chanels
    sepia_pixels = torch.zeros((3,256,256))
    r = dict["image"][0,:,:] #array of red values
    g = dict["image"][1,:,:] #array of green values
    b = dict["image"][2,:,:] # array of blue values
    
    #Transform original RGB channels to sepia tones
    tr = (0.393 * r + 0.769 * g + 0.189 * b)
    tg = (0.349 * r + 0.686 * g + 0.168 * b)
    tb = (0.272 * r + 0.534 * g + 0.131 * b) 
    
    #Threshold values:
    for x in range(0,256):
        for y in range(0,256):
            # Note we compare pixel values to 1 instead of 255 because these images have already been scaled in intensity
            if tr[x,y] > 1: 
                tr[x,y] = 1
            if tg[x,y] > 1:
                tg[x,y] = 1
            if tb[x,y] > 1:
                tb[x,y] = 1
                
    #Assign tensor of sepia values based on conversion above:
    for i in range(0,3):
        if i ==0:
            sepia_pixels[i,:,:] = tr
        elif i==1:
            sepia_pixels[i,:,:] = tg
        elif i==2:
            sepia_pixels[i,:,:] = tb

    #Create new dictionary, fill with sepia image and original label
    new_dict ={}
    new_dict["image"] = sepia_pixels #assign sepia values to image
    for key in dict.keys():
        if key != 'image':
            new_dict[key] = dict[key] #feed in original label value
            
    return new_dict


# Define pre_augmentation_transform to apply to data before applying augmentation_transforms:
pre_aug_transform = Compose(
    [   AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys="image"),
        Resized(keys=["image", "label"], spatial_size=(256,256), mode='nearest', align_corners=None),
        ToTensord(keys=["image", "label"])
    ]
)

# Define no_augmentation transform to apply to data in parallel to pre_aug_transform:
# this is in order to preserve the original, non-augmented images
# so that both augmented and non-augmented images are in the training set
no_augmentation_transforms = Compose(
    [   AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys="image"),
        Resized(keys = ["image", "label"], spatial_size=(256,256), mode='nearest', align_corners=None),
    ]
)

# Define augmentation transform types (given parameters in aug_params) which can be applied after pre_aug_transform:
def make_aug_transform(aug_params):
    dict = {
        'CustomGaussianNoised': add_noise,
        'RandAdjustContrastd': RandAdjustContrastd(keys=["image"], prob=aug_params['RandAdjustContrastd']['prob'], gamma=aug_params['RandAdjustContrastd']['gamma'], allow_missing_keys=aug_params['RandAdjustContrastd']['allow_missing_keys']),
        'GaussianSmoothd': GaussianSmoothd(keys=["image"], sigma=aug_params['GaussianSmoothd']['sigma']),
        'Rotated_180': Rotated(keys=["image", "label"], angle=aug_params['Rotated_180']['angle'], mode=aug_params['Rotated_180']['mode'] ),
        'RandZoomd': RandZoomd(keys=["image", "label"], prob=aug_params['RandZoomd']['prob'], min_zoom=aug_params['RandZoomd']['min_zoom'], max_zoom=aug_params['RandZoomd']['max_zoom'], padding_mode=NumpyPadMode.CONSTANT, mode=aug_params['RandZoomd']['mode']),
        'CenterSpatialCropd_200': CenterSpatialCropd(keys=["image", "label"], roi_size=aug_params['CenterSpatialCropd_200']['roi_size']),
        #'Sepia': sepia,
    } # note: The aug_params dictionary containing values for these transforms is defined in main_model_run.py
    return dict

# Define final transformation to apply to data after parallel aug_transform/no_augmentation_transform:
final_augmentation_transforms = Compose([
        Identityd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(256,256), mode='nearest', align_corners=None), #resize because of crop/zoom transforms
        ToTensord(keys=["image", "label"]),
        ToCategoricald,
    ]
)

# Define train transforms for case of no data augmentation
train_transforms = Compose(
    [   AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys="image"),
        Resized(keys=["image", "label"], spatial_size=(256,256), mode='nearest', align_corners=None),
        ToTensord(keys=["image", "label"]),
        ToCategoricald,
    ]
)

# Define validation transforms for validation dataset (no data augmentation):
val_transforms = Compose(
    [   AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys="image"),
        Resized(keys=["image", "label"], spatial_size=(256,256), mode='nearest', align_corners=None),
        ToTensord(keys=["image", "label"]),
        ToCategoricald,
    ]
)


# Function to save details of applied augmentation in csv within result_subfolder:
def save_aug_params(aug_params, aug_transformations, train_files, results_subfolder):
    
    #calculate number of augmented images created:
    num_augmented_data = len(train_files)*len(list(aug_transformations)) # assumes augmentation applied uniformly
    
    #create list of augmentations used and initialize dictionary:
    aug_list = list(aug_transformations)
    aug_img_count_dict = {}
    
    #create dictionary with augmentation type keys and the number of images with each augmentation:
    for aug_function in aug_list:
        aug_img_count_dict[aug_function] = len(train_files)
    print(aug_list)
    
    #initialize empty lists:
    aug_type_list = []
    num_img_list = []
    pct_img_list = []
    aug_params_list = []

    #for each augmentation type, populate the lists:
    for aug in aug_img_count_dict.keys():
        aug_type_list.append(aug)
        num_img_list.append(aug_img_count_dict[aug])
        pct_img_list.append(1) #100 perccent in this case because aug_img_count_dict[aug]/num_augmented_data = 1
        aug_params_list.append(aug_params[aug])

    #define dictionary to convert to pandas dataframe
    augmentation_params_dict = {
        'Aug_type': aug_type_list, 
        'num_imges': num_img_list, 
        'pct_img': pct_img_list,
        'params' : aug_params_list}

    #save augmentation parameters in pandas dataframe and save to csv:
    augmentation_params_df = pd.DataFrame.from_dict(augmentation_params_dict)
    augmentation_filename = 'AugParams_'+ str(num_augmented_data) + 'GeneratedImages.csv' 
    augmentation_params_df.to_csv(os.path.join(results_subfolder, augmentation_filename), index=False)
    
    
##################### Functions to handle out-of-distribution data types #########################
# Function converts nifti orientation to match expected image orientation:
def Nii2jpeg_orientationd(data_dict):
        temp_label = torch.fliplr(data_dict["label"])
        temp2_label = torch.rot90(temp_label, k=3, dims=[1,2])
        temp_image = torch.fliplr(data_dict["image"])
        temp2_image = torch.rot90(temp_image, k=3, dims=[1,2])
        data_dict["label"] = temp2_label
        data_dict["image"] = temp2_image
        return data_dict  


# Dictionary version of Grayscale transform:
def Image_to_Grayscale_d(data_dict):
    training_channels = 1
    img = data_dict["image"]
    transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                    torchvision.transforms.Grayscale(num_output_channels=training_channels),
                                    torchvision.transforms.ToTensor()
                                    ])
    grayscale_img = transform(img)
    new_dict = {"image": grayscale_img, "label": data_dict["label"]}
    return new_dict

# Convert labels from 1 channel with 5 class values to 5 channels with binary values (one-hot encoded), correct class labels for nifti images
def nifti_ToCategoricald(data_dict, n_classes=5):
        #flip to channel last:
        trans_label = torch.transpose(data_dict['label'],0,-1)
        
        #ensure class labels are consistent:
        new_label = torch.empty(trans_label.shape)
        new_label[:] = np.NaN
        for x in range(0, trans_label.shape[0]):
            for y in range(0, trans_label.shape[1]):
                value = int(trans_label[x,y,0])
                if value==1: #pink: anterior cervix + LUS
                    new_value = 2
                elif value==2: #blue: posterior cervix
                    new_value = 3
                elif value==3: #green: cervical canal + potential space between histological internal os and external os
                    new_value= 4
                elif value ==4: #yellow: bladder
                    new_value = 1
                elif value==0: #background
                    new_value = value
                new_label[x,y,0] = new_value
                         
        #add tensor dimension
        squeeze_label = torch.squeeze(new_label)
        
        #transform to one-hot encoded
        encoded_label = torch.from_numpy(keras.utils.to_categorical(squeeze_label.numpy(), num_classes=n_classes))
        
        #return categorical label in dictionary
        data_dict["label"] = torch.transpose(encoded_label,0,-1) #flip back to channel first

        return data_dict


# Transform which loads nifti datatypes:
nifti_transform = Compose(
    [   LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys="image"),
        Resized(keys = ["image", "label"], spatial_size =(256,256),mode='nearest', align_corners=None),
        ToTensord(keys=["image", "label"]),
        Image_to_Grayscale_d,
        Nii2jpeg_orientationd,
        nifti_ToCategoricald,
    ]
)