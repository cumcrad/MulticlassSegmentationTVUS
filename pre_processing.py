# Load packages:
import numpy as np
from my_transforms import *
import monai
from monai.data.utils import pad_list_data_collate
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


# This function takes a list of dictionaries containing paths of images and labels, and returns a list of dictionaries containing the
# image and label numpy arrays corresponding to each filename 
def convert_paths_to_np_preserveFileName(data_paths, training_channels):
    
    np_data_files_andNames = []
    for idx, dict in enumerate(data_paths):
        if training_channels == 1:
            #convert color image to gray scale while opening
            img = Image.open(dict["image"]).convert('L')
            img_np = np.array(img)
            #add a color channel back in (width, height, channel), last dimension
            img_np = np.expand_dims(img_np, -1)
        elif training_channels == 3:
            img = Image.open(dict["image"]) #open image
            img_np = np.array(img) # convert to numpy
        label = np.array(Image.open(dict["label"])) #open label, convert to numpy
        #create dictionary of file name, numpy array image, and numpy array label, append dictionary to list
        file_path = dict["image"]
        filename = file_path.split('/')[-1]
        np_data_files_andNames.append({"filename": filename, "image": img_np, "label": label})

    return np_data_files_andNames


# This function takes a list of dictionaries with {"filename", "image", "label"} keys and updates the label (numpy array)
# by converting pixel color assignments to class values
def ToClass_fnc_preserveFileName(data_dict_list):
    path_list = [data_dict_list[i]["filename"] for i in range(0,len(data_dict_list))]
    im_list = [data_dict_list[i]["image"] for i in range(0,len(data_dict_list))]
    lab_list = [data_dict_list[i]["label"] for i in range(0,len(data_dict_list))] 
    # Create a copy of the input list of dictionaries, for which the "label" values in the dictionaries will be updated:
    new_dict = [{"filename": path, "image": img, "label": label} for img, label, path in zip(im_list,lab_list, path_list)] 
    for i in range(0,len(data_dict_list)):
        input_array = data_dict_list[i]["label"] #take the labels only
        label_values = np.zeros((input_array.shape[0],input_array.shape[1]))
        new_data_4dict = np.zeros((input_array.shape[0],input_array.shape[1],1)) #define new dictionary to fill, only 1 channel needed
        for x in range(0,input_array.shape[0]):
            for y in range(0,input_array.shape[1]):
                #note - only the first RGB color channel was checked because the 5 label colors used contained unique values for the R channel
                if input_array[x,y,0] == 0: #class 0 color (0., 0., 0.) black
                    class_value = 0
                elif input_array[x,y,0] == 251: #class 1 color (251, 255, 28) yellow
                    class_value = 1
                elif input_array[x,y,0] == 255: #class 2 color (255, 52, 255) magenta
                    class_value = 2
                elif input_array[x,y,0] == 70: #class 3 color (70, 255, 232) cyan
                    class_value = 3
                elif input_array[x,y,0] == 75: #class 4 color (75, 232, 24) lime
                    class_value = 4
                #create matrix of class values 
                label_values[x,y] = class_value
                new_data_4dict[x,y,0] = class_value

        #reassign label values in new dictionary
        new_dict[i]["label"] = new_data_4dict
    return new_dict


# Given a list of images, segmentations and a grade (g6, g8, or g9), return a train, validation and test list of dictionaries (image/label pairs) with 70:20:10 split
# Within this function, images are read from paths, converted to class format, and divided into train/validation/test sets
def prep_data_split(image_id_list, segs, grade_string, training_channels, rand_state = 42):  
    grade_images = [] #initialize empty list of images corresponding to grade_string
    grade_segs = [] #initialize empty list of segmenations/labels corresponding to grade_string
    
    # note - there are more images than there are segmentations so we remove images with no mask pair
    # Only keep images for which there is a corresponding segmentation, whose filename contain the CLEAR grade_string (ex: 'g6', 'g8', or 'g9')
    for idx, seg in enumerate(segs):
        if grade_string in seg:           
            grade_images.append(image_id_list[idx])
            grade_segs.append(segs[idx])
            
    #create grouping of data paths, before splitting data set: (for grade_string only)
    grade_data_paths = [{"image": img, "label": seg} for img, seg in zip(grade_images,grade_segs)]
    #load numpy arrays from image paths
    grade_np_data_files = convert_paths_to_np_preserveFileName(grade_data_paths, training_channels) #outputs list of dictionaries with keys = {"filename", "image", "label"}
    #converts pixel color values to class values in segmentation mask - this function expects numpy inputs for keys = {"image", "label"}
    grade_class_json_dict = ToClass_fnc_preserveFileName(grade_np_data_files)
    #split the data into train, test and validation file lists: 70:20:10 split after applying train_test_split 2 times
    grade_train_files, grade_val_files = train_test_split(grade_class_json_dict, test_size = 0.3, random_state = rand_state)
    grade_val_files, grade_test_files = train_test_split(grade_val_files, test_size = 0.33, random_state = rand_state) 
    
    return grade_train_files, grade_val_files, grade_test_files


# Apply data augmentation (if used) on train_files and returns monai.data.DataLoader object for train, validation and test set
# as well as monai.data.CacheDataset for train, validation and test set
def load_data(train_files, val_files, test_files, batch_size, augmentation_flag, aug_transformations=None):
    
    if augmentation_flag == False:
    # create the training data loader
        train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
        train_loader = monai.data.DataLoader(train_ds, batch_size=batch_size, drop_last = False, shuffle=True, num_workers=4, collate_fn=pad_list_data_collate)
        
    elif augmentation_flag == True:     
        pre_aug_ds = pre_aug_transform(train_files)
        aug_ds = [] #initialize empty list to fill with augmented data

        #apply each augmentation one time to every image:
        for idx in range(0,len(aug_transformations)):
            aug_ds = []
            augmentation_key = list(aug_transformations)[idx]
            aug_transform = Compose([aug_transformations[augmentation_key]], [ToTensord(keys=["image", "label"])])
            aug_ds = aug_transform(pre_aug_ds)
            if (idx == 0):
                train_augmented_ds = pre_aug_ds + aug_ds
            else:
                train_augmented_ds = train_augmented_ds + aug_ds

        #remove unecessary keys from dictionary (image_transform, label_transform, image_meta..., label_meta...)
        light_train_augmented_ds = []
        for d in train_augmented_ds:
            light_train_augmented_ds.append({"image": d["image"], "label": d["label"]})

        # create the training data loader
        train_ds = monai.data.CacheDataset(data=light_train_augmented_ds, transform=final_augmentation_transforms, cache_rate=0.5)
        train_loader = monai.data.DataLoader(train_ds, batch_size=batch_size, drop_last = False, shuffle=True, num_workers=4, collate_fn=pad_list_data_collate)

    # create the validation data loader
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = monai.data.DataLoader(val_ds, batch_size=batch_size, drop_last = False, num_workers=4, collate_fn=pad_list_data_collate)

    # create the test data loader
    test_transforms = val_transforms
    test_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
    test_loader = monai.data.DataLoader(test_ds, batch_size=batch_size, drop_last = False, num_workers=4, collate_fn=pad_list_data_collate)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
