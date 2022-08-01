import cv2
import matplotlib.pyplot as plt
import torch
import os
from tensorflow import keras
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
import monai
import sklearn
from statsmodels.stats import inter_rater as irr
from PIL import Image
import my_transforms
from monai.data.utils import pad_list_data_collate


# Convert class dimension back to single dimension data:
def FromCategorical_toClass(seg_label, n_classes=5):
        #flip to channels last data:
        trans_label = seg_label
        #transform to numpy
        numpy_label = trans_label.numpy()
        #take argmax to reverse to_categorical
        arg_max_values = np.argmax(numpy_label, axis = 0)
        #convert to tensor
        class_values_tensor = torch.from_numpy(arg_max_values)
        
        return class_values_tensor


# Function to save transparent segmentation overlay on top of ultrasound image: 
def save_image_overlay(results_subfolder, files, n_classes):
    #iterate through all files:    
    for dict in files:
        #access image and filename
        filename = dict['filename']
        image = torch.from_numpy(dict['image']).to(torch.uint8)
        #switch to channel-first
        image = image.permute(2,0,1)
        #if image is grayscale (single channel), repeat color channel 3 times
        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
            
        #iterate through expert labels
        for key in dict['label'].keys():
            #access label from dictionary, convert to torch datatype
            label = dict['label'][key]
            label = torch.from_numpy(label)
            
            #convert to categorical
            trans_label = label.permute(2,0,1) #flip to channel last
            squeeze_label = torch.squeeze(trans_label) #add tensor dimension
            encoded_label = torch.from_numpy(keras.utils.to_categorical(squeeze_label.numpy(), num_classes=n_classes)) #transform to one-hot encoded
            label = encoded_label.permute(2,0,1) #flip back to channel first
            label = label.to(torch.bool) #convert to boolean data type
   
            #display label with transparency on top of image
            gt_segmentation_mask = torchvision.utils.draw_segmentation_masks(image, masks = label[1:], alpha = 0.6, colors = ["yellow", "magenta", "cyan", "lime" ])
            #convert to PIL image
            gt_img = F.to_pil_image(gt_segmentation_mask)

            #make subfolder, and save overlay images to folder
            InterOperator_folder = os.path.join(results_subfolder, 'Interoperator_Metrics')
            if not os.path.exists(str(InterOperator_folder)):
                os.mkdir(str(InterOperator_folder))
            img_folder = os.path.join(InterOperator_folder, 'Image_Overlay')
            if not os.path.exists(str(img_folder)):
                os.mkdir(str(img_folder))
            #define image name to save
            img_name = key + '_' + filename
            #write image to file
            cv2_img = cv2.cvtColor(np.asarray(gt_img), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(img_folder, img_name),cv2_img)
            
    return None


# Function to save metrics (Dice score, Hausdorff distance, Jaccard index) to evaluate inter-operator variability on CLEAR dataset:
def save_CLEAR_IOV_metrics(results_subfolder, KJ_test_ds, DC_test_ds, MH_test_ds, maj_test_ds, dice_metric, n_classes, test_filenames):
    
    #create dictionary of the test datasets from each expert - these should all be the same length
    test_ds = {"KJ": KJ_test_ds, "DC": DC_test_ds, "MH": MH_test_ds}
    
    #within the results subfolder, make an InterOperator_Metrics folder
    os.chdir(results_subfolder)
    if not os.path.exists("Interoperator_Metrics"):
        os.mkdir("Interoperator_Metrics")
    os.chdir("Interoperator_Metrics")
    IOV_metrics_results_subfolder = os.path.join(results_subfolder, 'Interoperator_Metrics')
    if not os.path.exists(str(IOV_metrics_results_subfolder)):
        os.mkdir(str(IOV_metrics_results_subfolder))
    
    #create empty dictionaries to store dice score, Hausdorff distance, and Jaccard index for single images, and all images across dataset:
    test_single_label_dice, test_single_label_HD, test_single_label_jaccard, test_all_data_dice, test_all_data_HD, test_all_data_jaccard = ({} for i in range(6))
    #create empty dictionaries to store class-specific average and standard deviation for each metric, across all images in test set
    test_across_data_dice, test_across_data_dice_std, test_across_data_HD, test_across_data_HD_std, test_across_data_jaccard, test_across_data_jaccard_std = ({} for i in range(6))
    #create empty dictionaries to store avgerage and standard deviation of AVERAGE dice score, Hausdorff distance, and Jaccard index across dataset:
    test_across_data_avg_dice, test_across_data_avg_dice_std, test_across_data_avg_HD, test_across_data_avg_HD_std, test_across_data_avg_jaccard, test_across_data_avg_jaccard_std = ({} for i in range(6))
    
    test_single_label_dice_avg, test_single_label_HD_avg, test_single_label_jaccard_avg, test_all_data_dice_avg, test_all_data_HD_avg, test_all_data_Jaccard_avg =  ({} for i in range(6)) 
    
    #initialize empty dictionaries with keys pointing to each expert
    for key in test_ds.keys():
        #fill initialized dictionaries with NaN values
        test_all_data_dice[key] = torch.empty((len(maj_test_ds),n_classes))
        test_all_data_dice[key][:] = float('nan')
        test_all_data_HD[key] = torch.empty((len(maj_test_ds),n_classes))
        test_all_data_HD[key][:] = float('nan')
        test_all_data_jaccard[key] = torch.empty((len(maj_test_ds),n_classes))
        test_all_data_jaccard[key][:] = float('nan')

    #fill dictionary with dice score, Hausdorff distance, and Jaccard index values for every image in the dataset
    with torch.no_grad():
        #access all image/label pairs in the majority GT dataset
        for idx, maj_image_pair in enumerate(maj_test_ds):
            #take the ground truth label from the majority vote segmentation for each image/label pair
            maj_gt_label = maj_image_pair["label"].detach().cpu().to(torch.bool)
            maj_gt_label = torch.unsqueeze(maj_gt_label, dim = 0)
            
            for key in test_ds.keys():         
                #load ground truth label for each expert, for corresponding image      
                image_pair = test_ds[key][idx] #image/segmentation pair
                gt_label = image_pair["label"].detach().cpu().to(torch.bool)
                gt_label = torch.unsqueeze(gt_label, dim = 0)
                
                #compute dice score for a single image
                test_single_label_dice[key] = dice_metric(y_pred=gt_label, y=maj_gt_label)
                
                #compute hausdorff distance for a single image
                test_single_label_HD[key] = monai.metrics.compute_hausdorff_distance(y_pred=gt_label, y=maj_gt_label, include_background = True)
                
                #compute jaccard index for a single image:          
                test_single_label_jaccard[key] = sklearn.metrics.jaccard_score(y_true = maj_gt_label.flatten(2,3).squeeze(0).swapaxes(0,1), y_pred = gt_label.flatten(2,3).squeeze(0).swapaxes(0,1), average = None)           
                test_single_label_jaccard[key] = torch.from_numpy(test_single_label_jaccard[key]).unsqueeze(0)
                    
                #calculate average dice metric, hausdorff distance, and jaccard index for each image:
                test_single_label_dice_avg[key] = torch.mean(test_single_label_dice[key]).unsqueeze(0).unsqueeze(0)
                test_single_label_HD_avg[key] = torch.mean(test_single_label_HD[key]).unsqueeze(0).unsqueeze(0)
                test_single_label_jaccard_avg[key] = torch.mean(test_single_label_jaccard[key]).unsqueeze(0).unsqueeze(0)
                
                #generate tensor of dice score, hausdorff distance, and jaccard index for all images in test set
                if idx==0:
                    test_all_data_dice[key] = test_single_label_dice[key]
                    test_all_data_HD[key] = test_single_label_HD[key]
                    test_all_data_jaccard[key] = test_single_label_jaccard[key]
                    test_all_data_dice_avg[key] = test_single_label_dice_avg[key]
                    test_all_data_HD_avg[key] = test_single_label_HD_avg[key]
                    test_all_data_Jaccard_avg[key] = test_single_label_jaccard_avg[key]
                else:
                    #concatenate class-specific dice metric, hausdorff distance, and jaccard index along image dimension
                    test_all_data_dice[key] = torch.cat((test_all_data_dice[key], test_single_label_dice[key]), dim = 0)
                    test_all_data_HD[key] = torch.cat((test_all_data_HD[key], test_single_label_HD[key]), dim=0)
                    test_all_data_jaccard[key] = torch.cat((test_all_data_jaccard[key], test_single_label_jaccard[key]), dim=0)
                    #concatenate average dice metric, hausdorff distance, and jaccard index (averaged across class dimension for a single image) along image dimension
                    test_all_data_dice_avg[key] = torch.cat((test_all_data_dice_avg[key], test_single_label_dice_avg[key]) , dim = 0)
                    test_all_data_HD_avg[key] = torch.cat((test_all_data_HD_avg[key], test_single_label_HD_avg[key]) , dim = 0)
                    test_all_data_Jaccard_avg[key] = torch.cat((test_all_data_Jaccard_avg[key], test_single_label_jaccard_avg[key]) , dim = 0)
                    
                #reset dice metric    
                dice_metric.reset()
        
        #calculate average values across all images in the dataset:
        for key in test_ds.keys():
            
            #calculate class-specific average and standard deviation for each metric, across all images in test set
            #each of the 6 variables below should be of dimension 1-by-#classes, in this case [1,5]
            test_across_data_dice[key] = torch.nanmean(test_all_data_dice[key], dim=0)
            test_across_data_dice_std[key] = torch.std(test_all_data_dice[key], dim=0)
            test_across_data_HD[key] = torch.nanmean(test_all_data_HD[key], dim=0)
            test_across_data_HD_std[key] = torch.std(test_all_data_HD[key], dim=0)
            test_across_data_jaccard[key] = torch.nanmean(test_all_data_jaccard[key], dim =0)
            test_across_data_jaccard_std[key] = torch.std(test_all_data_jaccard[key], dim = 0)
   
            #calculate average and standard deviation (of class-average dice metric, hausdorff distance, and jaccard index) across all images in test set
            test_across_data_avg_dice[key] = torch.nanmean(test_all_data_dice_avg[key], dim = 0)
            test_across_data_avg_dice_std[key] = torch.std(test_all_data_dice_avg[key], dim = 0)
            test_across_data_avg_HD[key] = torch.nanmean(test_all_data_HD_avg[key], dim = 0)
            test_across_data_avg_HD_std[key] = torch.std(test_all_data_HD_avg[key], dim = 0)
            test_across_data_avg_jaccard[key] = torch.nanmean(test_all_data_Jaccard_avg[key], dim = 0)
            test_across_data_avg_jaccard_std[key] = torch.std(test_all_data_Jaccard_avg[key], dim = 0)
        
        # Save test metrics to csv file:
        metric_names = ['Dice', 'Dice_std', 'HD', 'HD_std', 'Jaccard', 'Jaccard_std' ]
        
        average_metrics = {"KJ": [test_across_data_avg_dice["KJ"].item(), test_across_data_avg_dice_std["KJ"].item(),
                        test_across_data_avg_HD["KJ"].item(), test_across_data_avg_HD_std["KJ"].item(),
                        test_across_data_avg_jaccard["KJ"].item(), test_across_data_avg_jaccard_std["KJ"].item()],
                           
                           "DC": [test_across_data_avg_dice["DC"].item(), test_across_data_avg_dice_std["DC"].item(),
                        test_across_data_avg_HD["DC"].item(), test_across_data_avg_HD_std["DC"].item(),
                        test_across_data_avg_jaccard["DC"].item(), test_across_data_avg_jaccard_std["DC"].item()],
                           
                           "MH": [test_across_data_avg_dice["MH"].item(), test_across_data_avg_dice_std["MH"].item(),
                        test_across_data_avg_HD["MH"].item(), test_across_data_avg_HD_std["MH"].item(),
                        test_across_data_avg_jaccard["MH"].item(), test_across_data_avg_jaccard_std["MH"].item()],
                           }
        
        all_classes_metrics = []
        for class_dim in range(0,n_classes):
            class_metrics = {}
            for key in test_ds.keys():
                class_metrics[key] = [test_across_data_dice[key][class_dim].item(), test_across_data_dice_std[key][class_dim].item(),
                                      test_across_data_HD[key][class_dim].item(), test_across_data_HD_std[key][class_dim].item(),
                                      test_across_data_jaccard[key][class_dim].item(), test_across_data_jaccard_std[key][class_dim].item()] 
            all_classes_metrics.append(class_metrics)

        background_metrics = all_classes_metrics[0]
        bladder_metrics = all_classes_metrics[1]
        anterior_metrics = all_classes_metrics[2]
        posterior_metrics = all_classes_metrics[3]
        canal_metrics = all_classes_metrics[4]
        
        test_df_KJ = pd.DataFrame({'Metric Name': metric_names, 'avg': average_metrics['KJ'], 'background': background_metrics['KJ'], 'bladder': bladder_metrics['KJ'], 'anterior': anterior_metrics['KJ'], 'posterior': posterior_metrics['KJ'], 'canal': canal_metrics['KJ']})
        test_df_KJ.to_csv(os.path.join(IOV_metrics_results_subfolder,'Test_Metric_variables_KJ.csv'), index=False)

        test_df_DC = pd.DataFrame({'Metric Name': metric_names, 'avg': average_metrics['DC'], 'background': background_metrics['DC'], 'bladder': bladder_metrics['DC'], 'anterior': anterior_metrics['DC'], 'posterior': posterior_metrics['DC'], 'canal': canal_metrics['DC']})
        test_df_DC.to_csv(os.path.join(IOV_metrics_results_subfolder,'Test_Metric_variables_DC.csv'), index=False)
    
        test_df_MH = pd.DataFrame({'Metric Name': metric_names, 'avg': average_metrics['MH'], 'background': background_metrics['MH'], 'bladder': bladder_metrics['MH'], 'anterior': anterior_metrics['MH'], 'posterior': posterior_metrics['MH'], 'canal': canal_metrics['MH']})
        test_df_MH.to_csv(os.path.join(IOV_metrics_results_subfolder,'Test_Metric_variables_MH.csv'), index=False)
        
        #save detailed dice scores, hausdorff distance, jaccard index and name of corresponding image (for each image in the test set) to csv
        metrics = ['Dice', 'HD', 'JI']
        test_all_data = {'Dice': test_all_data_dice, 'HD': test_all_data_HD, 'JI': test_all_data_jaccard}
        
        for metric in metrics:
            avg_col, background_col, bladder_col, anterior_col, posterior_col, canal_col = ({"KJ":[], "DC": [], "MH": []} for i in range(6))  
            for key in test_ds.keys():
                for i in range(len(test_filenames)):
                    avg_col[key].append(torch.nanmean(test_all_data[metric][key][i][1:-1]).item()) #take average of all classes except background
                    background_col[key].append(test_all_data[metric][key][i][0].item())
                    bladder_col[key].append(test_all_data[metric][key][i][1].item())
                    anterior_col[key].append(test_all_data[metric][key][i][2].item())
                    posterior_col[key].append(test_all_data[metric][key][i][3].item())
                    canal_col[key].append(test_all_data[metric][key][i][4].item())
            
                avgNoBackground_metric_name = 'avg_' + metric + 'noBackground'
                background_metric_name = 'background_' + metric
                bladder_metric_name = 'bladder' + metric
                anterior_metric_name = 'anterior_' + metric
                posterior_metric_name = 'posterior_' + metric
                canal_metric_name = 'canal_' + metric
                test_df_each_image_expert = pd.DataFrame({'Filename': test_filenames, avgNoBackground_metric_name: avg_col[key],
                            background_metric_name: background_col[key], bladder_metric_name: bladder_col[key],
                            anterior_metric_name: anterior_col[key], posterior_metric_name: posterior_col[key],
                            canal_metric_name: canal_col[key]})
                allImage_metric_filename = 'Test_AllImages_' + metric + '_' + key + '.csv'
                test_df_each_image_expert.to_csv(os.path.join(IOV_metrics_results_subfolder, allImage_metric_filename), index=False)

    return None


# Function to calculate Fliess Kappa score, returns average fliess kappa calculated across all images
def fliess_kappa(results_subfolder, KJ_test_loader, DC_test_loader, MH_test_loader, test_filenames):
    
    test_loaders = {'KJ': KJ_test_loader, 'DC': DC_test_loader, 'MH': MH_test_loader}
    class_values = {'KJ': [], 'DC': [], 'MH': []}
    expert_segs = {'KJ': [], 'DC': [], 'MH': []}
    all_images_fk = []
    
    #access batched data for all 3 experts, convert to class values, concatenate along image dimension
    for key in expert_segs.keys():
        for batch_data in test_loaders[key]:
            for idx in range(0,batch_data["label"].shape[0]):
                class_values[key].append(FromCategorical_toClass(batch_data["label"][idx]))
        for i, seg in enumerate(class_values[key]):
            if i == 0:
                expert_segs[key] = torch.flatten(seg, start_dim = 0, end_dim = -1).unsqueeze(0)
            if i > 0:
                expert_segs[key] = torch.cat((expert_segs[key], torch.flatten(seg, start_dim = 0, end_dim = -1).unsqueeze(0)), dim=0)

    #concatenate along the expert label dimensions    
    all_experts_segs =  torch.cat((expert_segs["KJ"].unsqueeze(1), expert_segs["DC"].unsqueeze(1)), dim = 1)
    all_experts_segs = torch.cat((all_experts_segs, expert_segs["MH"].unsqueeze(1)), dim = 1)

    #iterate through every image to calculate Fliess Kappa score
    for i in range(0,all_experts_segs.shape[0]):
        columns_experts = np.array(all_experts_segs[i]).transpose()
        count_all_expert_class_values, class_categories = irr.aggregate_raters(columns_experts)
        #calculate fleiss kappa
        image_fk = irr.fleiss_kappa(count_all_expert_class_values, method ='fleiss')
        print('fleiss kappa score for image {} is {}'.format(i, image_fk))
        #append values to list
        all_images_fk.append(image_fk)

    #calculate average Fliess Kappa score, across all images
    avg_fk = np.mean(all_images_fk)
    print("average fleiss kappa is {}".format(avg_fk))
        
    #save results to InterOperator_Metrics folder (within results subfolder)
    os.chdir(results_subfolder)
    if not os.path.exists("Interoperator_Metrics"):
        os.mkdir("Interoperator_Metrics")
    os.chdir("Interoperator_Metrics")
    IOV_metrics_results_subfolder = os.path.join(results_subfolder, 'Interoperator_Metrics')
    if not os.path.exists(str(IOV_metrics_results_subfolder)):
        os.mkdir(str(IOV_metrics_results_subfolder))
    
    FK_eachImage = pd.DataFrame({'filename': test_filenames, 'Fliess Kappa': all_images_fk})
    FK_eachImage.to_csv(os.path.join(IOV_metrics_results_subfolder,'Test_AllImages_FliessKappa.csv'), index=False)
    
    FK_avg = pd.DataFrame({'Avg Fliess Kappa': [avg_fk]})
    FK_avg.to_csv(os.path.join(IOV_metrics_results_subfolder,'Test_AvgFliessKappa.csv'), index=False)
    
    return avg_fk


# Interoperator-version of convert_paths_to_np_preserveFileNames() from pre_processing.py
# This version expects takes data paths and splits them into a dictionary with the expert labelers
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
            img_np = np.array(img) #convert to numpy
        #create label dictionary containing the np array for each expert label
        KJ_label = np.array(Image.open(dict["label"]["KJ"]))
        DC_label = np.array(Image.open(dict["label"]["DC"]))
        MH_label = np.array(Image.open(dict["label"]["MH"]))
        maj_label = np.array(Image.open(dict["label"]["maj"]))
        label_dict = {"KJ": KJ_label, "DC": DC_label, "MH": MH_label, "maj": maj_label}
        #create dictionary of with filenames, images, and label sub-dictionary
        file_path = dict["image"]
        filename = file_path.split('/')[-1]
        np_data_files_andNames.append({"filename": filename, "image": img_np, "label": label_dict})
        
    return np_data_files_andNames


# Interoperator-version of ToClass_fnc_preserveFileName() from pre_processing.py
# This version expects elements of data_dict to have keys corresponding to experts
def ToClass_fnc_preserveFileName(data_dict):
    path_list = [data_dict[i]["filename"] for i in range(0,len(data_dict))]
    im_list = [data_dict[i]["image"] for i in range(0,len(data_dict))]
    lab_dict = [data_dict[i]["label"] for i in range(0,len(data_dict))]
    # Create a copy of the input list of dictionaries, for which the "label" values in the dictionaries will be updated:
    new_dict = [{"filename": path, "image": img, "label": label} for img, label, path in zip(im_list, lab_dict, path_list)]
    
    for i in range(0,len(data_dict)):
        for key in data_dict[i]["label"].keys():
            input_array = data_dict[i]["label"][key] #take the labels only
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
            new_dict[i]["label"][key] = new_data_4dict
            
    return new_dict

# Interoperator-version of load_data() from preprocessing.py
# This version creates a data loader for each expert
def load_all_data(KJ_all_files_data, DC_all_files_data, MH_all_files_data, maj_all_files_data, batch_size, augmentation_flag, aug_transformations=None):
    
    # load the entire dataset for each expert
    all_KJ_ds = monai.data.CacheDataset(data=KJ_all_files_data, transform=my_transforms.val_transforms, cache_rate=1.0)#
    all_KJ_loader = monai.data.DataLoader(all_KJ_ds, batch_size=batch_size, drop_last = False, num_workers=4, collate_fn=pad_list_data_collate)

    all_DC_ds = monai.data.CacheDataset(data=DC_all_files_data, transform=my_transforms.val_transforms, cache_rate=1.0)#
    all_DC_loader = monai.data.DataLoader(all_DC_ds, batch_size=batch_size, drop_last = False, num_workers=4, collate_fn=pad_list_data_collate)

    all_MH_ds = monai.data.CacheDataset(data=MH_all_files_data, transform=my_transforms.val_transforms, cache_rate=1.0)#
    all_MH_loader = monai.data.DataLoader(all_MH_ds, batch_size=batch_size, drop_last = False, num_workers=4, collate_fn=pad_list_data_collate)

    all_maj_ds = monai.data.CacheDataset(data=maj_all_files_data, transform=my_transforms.val_transforms, cache_rate=1.0)#
    all_maj_loader = monai.data.DataLoader(all_maj_ds, batch_size=batch_size, drop_last = False, num_workers=4, collate_fn=pad_list_data_collate)

    return all_KJ_loader, all_DC_loader, all_MH_loader, all_maj_loader, all_KJ_ds, all_DC_ds, all_MH_ds, all_maj_ds
