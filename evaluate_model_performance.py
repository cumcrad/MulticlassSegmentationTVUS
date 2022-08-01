# Load packages:
import sys
import torch
import numpy as np
import os
from PIL import Image
import monai
from monai.metrics import DiceMetric
from glob import glob
from monai.transforms import( AsChannelFirstd, Compose, Resized, ScaleIntensityd, ToTensord)
#import custom methods/functions:
import Inpaint_func
import save_model_metrics
import my_transforms



# convert images only to numpy datatype - this is for unlabaled data
def convert_img_paths_to_np(data_paths, training_channels):
    np_data_files = []
    for idx, dict in enumerate(data_paths):
        if training_channels == 1:
            #convert color image to gray scale while opening
            img = Image.open(dict["image"]).convert('L')
            img_np = np.array(img)
            #add a color channel back in (width, height, channel), last dimension
            img_np = np.expand_dims(img_np, -1)
        elif training_channels == 3:
            #open image, convert to numpy
            img = Image.open(dict["image"])
            img_np = np.array(img)
        np_data_files.append({"image": img_np})
    return np_data_files


def model_predict_eval(dataset_type, device_number):

    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Set model folder to access:
    model_folder = '/home/obsegment/code/ResearchDataset/CLEAR/Results/20220628_UNet_ch1Res4_Inpaint_optimAdam_lr0p001_dropoutp2_batchsz16_epoch50_patience5_BestValDice_augRcGsGnRtCrRz_numAug1026'
    # ---------------------------------------------------------------------------------------------------------------------------- #

    #load UNet model:
    training_channels = 1 #channels to train UNet on (subset)
    dice_metric = DiceMetric(include_background=True, reduction="mean") #dice metric to evaluate model performance during training

    os.chdir(model_folder)
    print("current working directory is {}".format(os.getcwd()))
    
    #read model parameters from model_folder name
    num_res_units = int(model_folder.split('/')[-1].split('Res')[-1][0]) #4
    dropout =  float('0.' + model_folder.split('/')[-1].split('dropoutp')[-1][0]) #0.2
    print("num_res_units = {}".format(num_res_units))
    print("dropout = {}".format(dropout))
    device = int(device_number)
    #model definition
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=training_channels,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2), 
        num_res_units=num_res_units, 
        dropout=dropout 
    ).to(device)
    #load hyperparameters:
    model.load_state_dict(torch.load("best_metric_model_segmentation2d_dict.pth"))
    model.eval()
    
    if dataset_type == 'Bounce':
        #folder of out-of-distribution images for testing:
        evaluate_images_folder = 'Bounce/16weeks'
        evaluate_images_path = os.path.join('/home/obsegment/code/ResearchDataset/', evaluate_images_folder)
        img_prefix = '*BM*'
        
        #define transforms for this dataset
        datatype_transform = Compose(
            [   
                AsChannelFirstd(keys=["image"], channel_dim=-1),
                ScaleIntensityd(keys="image"),
                Resized(keys = ["image"], spatial_size =(256,256),mode='nearest', align_corners=None),
                ToTensord(keys=["image"])
            ]
        )
        
        os.chdir(evaluate_images_path)   
        # Open image/labels as numpy arrays, convert label to class value     
        images = sorted(glob(os.path.join(evaluate_images_path, img_prefix)))
        data_paths = [{"image": img} for img in images]
        filenames = [image.split('/')[-1].replace('.tif','.png') for image in images] #image name including png extension
        #load data:
        np_data_files = convert_img_paths_to_np(data_paths, training_channels)
        class_json_dict = np_data_files
        ds = monai.data.CacheDataset(data=class_json_dict, transform=datatype_transform, cache_rate=0.5)
        
    elif dataset_type == "LORI":
        #folder of out-of-distribution images for testing:
        evaluate_images_folder = 'LORI'
        evaluate_images_path = os.path.join('/home/obsegment/code/ResearchDataset/', evaluate_images_folder)
        if not os.path.exists(evaluate_images_path):
            os.mkdir(evaluate_images_path)
        img_prefix = "*.jpg"
        #define transforms for this dataset
        datatype_transform = my_transforms.nifti_transform
        
        os.chdir(evaluate_images_path)   
        # Open image/labels as numpy arrays, convert label to class value     
        images = sorted(glob(os.path.join(evaluate_images_path, img_prefix)))
        masks = sorted(glob(os.path.join(evaluate_images_path,"*.nii.gz")) + glob(os.path.join(evaluate_images_path, "*.nii")))
        data_paths = [{"image": img, "label": seg} for img, seg in zip(images, masks)]
        filenames =[image.split('/')[-1].replace('.jpg','.png') for image in images] #image name including png extension
        print("filenames = {}".format(filenames))
        
        #load data:
        ds = monai.data.CacheDataset(data=data_paths, transform=datatype_transform, cache_rate=0.5)
    
    elif dataset_type == "ATOPS":
        #folder of out-of-distribution images for testing:
        evaluate_images_folder = 'ATOPS'
        evaluate_images_path = os.path.join('/home/obsegment/code/ResearchDataset/', evaluate_images_folder)
        if not os.path.exists(evaluate_images_path):
            os.mkdir(evaluate_images_path)
        img_prefix = "*.jpg"
        #define transforms for this dataset
        datatype_transform = my_transforms.nifti_transform
        
        os.chdir(evaluate_images_path)   
        # Open image/labels as numpy arrays, convert label to class value     
        images = sorted(glob(os.path.join(evaluate_images_path, img_prefix)))
        masks = sorted(glob(os.path.join(evaluate_images_path,"*.nii.gz")) + glob(os.path.join(evaluate_images_path, "*.nii")))
        data_paths = [{"image": img, "label": seg} for img, seg in zip(images, masks)]
        filenames =[image.split('/')[-1].replace('.jpg','.png') for image in images] #image name including png extension
        print("filenames = {}".format(filenames))

        #load data:
        ds = monai.data.CacheDataset(data=data_paths, transform=datatype_transform, cache_rate=0.5)
        
    #evaluate model performance on test set, save images
    save_model_metrics.save_prediction_images_and_metrics(evaluate_images_path, model, ds, filenames, device, dice_metric, dataset_type, out_of_distribution = True)
 
if __name__ == '__main__':
    # Map command line arguments to function arguments
    model_predict_eval(*sys.argv[1:])
    
    


