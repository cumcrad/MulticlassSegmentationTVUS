# Load packages:
import os
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from model_utils import batched_post_trans
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import monai
from tensorflow import keras
from my_transforms import ToCategoricald
import sklearn
from sklearn.metrics import f1_score
from monai.transforms import( AsChannelFirstd, Compose, LoadImaged, Resized, ScaleIntensityd, ToTensord)

# Save record of filenames within train, validation and test set to csv in input results_subfolder:
def save_image_filenames(train_filenames, val_filenames, test_filenames, results_subfolder):
    
    #create variable with extended length, to match length of train_filenames:
    num_train_images = len(train_filenames)
    extended_val_filenames = [None] * num_train_images 
    extended_test_filenames = [None] * num_train_images
    extended_val_filenames[0:len(val_filenames)] = val_filenames
    extended_test_filenames[0:len(test_filenames)] = test_filenames
    
    #create dictionary, convert to dataframe, save as csv
    filename_dict = {"train_images": train_filenames, "val_images": extended_val_filenames, "test_images": extended_test_filenames}
    filename_df = pd.DataFrame(filename_dict)
    filename_df.to_csv(os.path.join(results_subfolder,'TrainValTest_ImageNames.csv'), index=False)


# Define name of folder to store all results and trained model:
# This name includes the hyperparameter values and date of run, to uniquely identify the model results
def create_model_run_folder(results_folder, aug_type, n_train_files, n_val_files, n_test_files, n_aug_transformations, num_res_units,
lr_stepsize, lr_gamma, training_channels, optim, batch_size, dropout, n_epochs, use_AMF_images, use_Inpaint_images, model_type, es_patience):

    #create results subfolder based on input results_folder
    os.chdir(results_folder)
    if not os.path.exists(str(results_folder)):
        os.mkdir(str(results_folder))

    #retrieve today's date, will be used as part of the subfolder path
    run_date = str(date.today()).replace('-', '')
    
    #number of augmented images (assuming each form of data augmentation is applied once to every image), will be used as part of subfolder path
    num_augmented_data = n_train_files*n_aug_transformations

    #convert learning rate and dropout values to string, will be used as part of subfolder path
    lr = str(lr_stepsize) + 'p' + str(lr_gamma)[2:]
    drop = str(dropout)[2:]

    #define string to indicate preprocessing type, will be used as part of subfolder path
    if use_AMF_images:
        preprocess_type = '_AMF'
    elif use_Inpaint_images:
        preprocess_type = '_Inpaint'
    else:
        preprocess_type = '_Orig'
        
    #make subfolder for model run:    
    if model_type == 'UNet':
        results_subfolder_str = run_date + '_' + model_type + '_ch' + str(training_channels) + 'Res' + str(num_res_units) +  preprocess_type + '_optim' + optim + '_'+ 'lr' +lr + '_dropoutp' + str(drop) + '_batchsz' + str(batch_size) + '_epoch'+ str(n_epochs)  + '_patience' + str(es_patience) + '_BestValDice' + '_aug' + aug_type
    else: #model_type == 'AttentionUNet', 'UNETR', or 'SegResNet'
        results_subfolder_str = run_date + '_' + model_type + '_ch' + str(training_channels) +  preprocess_type + '_optim' + optim + '_'+ 'lr' +lr + '_dropoutp' + str(drop) + '_batchsz' + str(batch_size) + '_epoch'+ str(n_epochs)  + '_patience' + str(es_patience) + '_BestValDice' + '_aug' + aug_type
    if aug_type != "None": #add number of augmented images to subfolder string
         results_subfolder_str = results_subfolder_str + '_numAug' + str(num_augmented_data)
        
    #create results subfolder
    results_subfolder = os.path.join(results_folder, results_subfolder_str)
    if not os.path.exists(results_subfolder): #
        os.mkdir(results_subfolder)

    #define model name
    model_name = 'Model.pt'

    #save number of images within train, validation, and testing to csv:
    dataset_size = pd.DataFrame({'training_set_size': n_train_files, 'validation_set_size': n_val_files, 'test_set_size': n_test_files}, index=[0])
    dataset_size.to_csv(os.path.join(results_subfolder,'DatasetSizeDivisions.csv'), index=False)

    return model_name, results_subfolder


# Create plots of loss and dice metric vs #epochs, save within results_subfolder:
def save_loss_dice_plots(train_loss, val_loss, train_meanDICE, val_meanDICE, results_subfolder, best_epoch):

    #generate list of epochs, to be used as x-axis:
    epochs = [item for item in range(1,len(train_loss)+1)]
    
    #create dice and loss plots, side-by-side
    dice_loss_plot = plt.figure("label", (18,7))
    font = {'size':20}
    plt.rc('font', **font)
    plt.rc('xtick', labelsize = 20) 
    plt.rc('ytick', labelsize = 20)
    plt.rcParams['lines.linewidth'] = 3
    
    for i in range(2): #mean DICE score vs. # epochs
        plt.subplot(1,2,i+1)
        if i==0:
            plt.plot(epochs, train_meanDICE, label = "train")
            plt.plot(epochs, val_meanDICE, label = "validation")
            plt.xlabel("Epochs")
            plt.ylabel("mean DICE coefficient")
            plt.axvline(best_epoch, linestyle = '--', color = 'r', label = 'model saved') #line indicates epoch when model saved (which had best validation dice score)
            plt.legend(fontsize = 20) 
        if i==1: #loss vs. # epochs
            plt.plot(epochs,train_loss, label = "train")
            plt.plot(epochs,val_loss, label = "validation")
            plt.xlabel("Epochs")
            plt.ylabel("loss")
            plt.axvline(best_epoch, linestyle = '--', color = 'r', label = 'model saved') #line indicates epoch when model saved (which had best validation dice score)
            plt.legend(fontsize = 20) 
    plt.savefig(results_subfolder + '/DiceLossEpochPlot.png')
    plt.close()

    #save dice and loss plots, individually as separate figures
    for i in range(2):
        if i==0: # mean dice score vs # epochs
            diceplt= plt.figure("label", (9,6))
            plt.rc('font', **font)
            plt.rc('xtick', labelsize = 20)
            plt.rc('ytick', labelsize = 20)
            plt.rcParams['lines.linewidth'] = 3
            plt.plot(train_meanDICE, label = "train")
            plt.plot(val_meanDICE, label = "validation")
            plt.xlabel("Epochs")
            plt.ylabel("mean DICE coefficient")
            plt.axvline(best_epoch, linestyle = '--', color = 'r', label = 'model saved') #line indicates epoch when model saved (which had best validation dice score)
            plt.legend(fontsize = 20)
            plt.savefig(results_subfolder + '/DiceEpochPlot.png')
            plt.close()
        if i==1: #loss vs. # epochs
            loss_plt = plt.figure("label", (9,6))
            plt.rc('font', **font)
            plt.plot(train_loss, label = "train")
            plt.plot(val_loss, label = "validation")
            plt.xlabel("Epochs")
            plt.ylabel("loss")
            plt.axvline(best_epoch, linestyle = '--', color = 'r', label = 'model saved') #line indicates epoch when model saved (which had best validation dice score)
            plt.legend(fontsize = 20)     
            plt.savefig(results_subfolder + '/LossEpochPlot.png')
            plt.close()


# Create plots of loss and dice metrics vs #epochs, plot separate lines for each segmentation class, save within results_subfolder:
def save_class_dice_plots(val_meanDICE, train_dice_df, val_dice_df, results_subfolder, best_epoch):
    
    font = {'size': 20}
    
    #generate class names list based on columns of val/train dice dataframe
    val_dice_cols = val_dice_df.columns.to_list()
    train_dice_cols = train_dice_df.columns.to_list()
    saved_epoch = best_epoch

    #create dictionary storing class names and colors
    class_names_colors = {
        'background_dice': {'title':'Background', 'color': 'black'},
        'bladder_dice': {'title':'Bladder', 'color': 'orange'}, #orange instead of yellow to create readable graph
        'anterior_dice': {'title':'Anterior Cervix', 'color': 'magenta'},
        'posterior_dice': {'title':'Posterior Cervix', 'color': 'cyan'},
        'potential_space_dice': {'title':'Cervical Canal', 'color': 'lime'}
    }
    
    #save individual plots with validation and train dice scores, one plot per class
    for idx in range(1,len(val_dice_cols)):
        train_dice_col = train_dice_cols[idx]
        plot_title = class_names_colors[train_dice_col]['title']
        class_train_dice = train_dice_df[train_dice_col].tolist()
        val_dice_col = val_dice_cols[idx]
        class_val_dice = val_dice_df[val_dice_col].tolist()
        diceplt = plt.figure("label", (9,6))
        plt.rc('font', **font)
        plt.title(plot_title)
        plt.plot(class_train_dice, label = "train")
        plt.plot(class_val_dice, label = "validation") 
        plt.axvline(saved_epoch, linestyle = '--', color = 'r', label = 'model saved') #line indicates epoch when model saved (which had best validation dice score)
        plt.xlabel("Epochs")
        plt.ylabel("mean DICE coefficient")
        plt.legend()
        plt.savefig(results_subfolder + '/' + val_dice_col + '_DiceEpochPlot.png')
        plt.close()

    #save a single graph, plotting validation dice values for each class
    classes_diceplt = plt.figure("label",(12,8))
    plt.rc('font', **font)
    plt.rc('xtick', labelsize = 20)
    plt.rc('ytick', labelsize = 20)
    plt.rcParams['lines.linewidth'] = 3 
    for idx in range(1,len(val_dice_cols)):
        train_dice_col = train_dice_cols[idx]
        legend_key = class_names_colors[train_dice_col]['title']
        class_color = class_names_colors[train_dice_col]['color']
        class_train_dice = train_dice_df[train_dice_col].tolist()
        val_dice_col = val_dice_cols[idx]
        class_val_dice = val_dice_df[val_dice_col].tolist()
        plt.plot(class_val_dice, label = legend_key, c = class_color)
    plt.plot(val_meanDICE, label = 'Average', c = 'grey', linestyle = '--')
    plt.axvline(saved_epoch, linestyle = '--', color = 'r', label = 'model saved') #line indicates epoch when model saved (which had best validation dice score)
    plt.xlabel("Epochs")
    plt.ylabel("mean DICE coefficient")
    plt.legend(fontsize = 24)
    plt.savefig(results_subfolder + '/ValClass_DiceEpochPlot.png')
    plt.close()


# Display and save by side-by-side image comparing ground truth segmentation to predicted segmentation:
# Segmentation is displayed with transparency on top of the original input image
def display_gt_vs_prediction_mask(gt_imgs, pred_imgs, subset_results_subfolder, filenames, dataset_type, out_of_distribution = False):
    
    plt.rcParams["savefig.bbox"] = 'tight'
    
    for i, images in enumerate(gt_imgs):
        compare_masks= plt.figure()
        filename = filenames[i]
        
        for idx in range(0,2):
            plt.subplot(1,2,idx+1)
            plt.tight_layout()
            plt.tick_params(labelleft=False, labelbottom=False)
            ax= plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            
            #ground truth segmentation overlay
            if idx==0:
                gt_img = gt_imgs[i].detach()
                gt_img = F.to_pil_image(gt_img)
                plt.imshow(np.asarray(gt_img))
                gt_img = gt_img.resize((512,512))
                plt.title('Ground Truth')
            
            #prediction segmentation overlay    
            if idx==1:
                pred_img = pred_imgs[i].detach()
                pred_img = F.to_pil_image(pred_img)
                pred_img = pred_img.resize((512,512))
                plt.title("Prediction")
                plt.imshow(np.asarray(pred_img))
                
            #make subfolder, save figure to file
            img_folder = os.path.join(subset_results_subfolder, 'ModelPredictionImages')
            if not os.path.exists(str(img_folder)):
                os.mkdir(str(img_folder))
            
            #set prefix for file naming based on which dataset is being used
            if out_of_distribution:
                img_name = filename
            else:
                img_name = dataset_type.capitalize() + '_' + filename    
            plt.savefig(os.path.join(img_folder, img_name))
        plt.close()


# Utility function used in save_val_prediction_images_and_metrics() to save images that provide some insight into how the model makes its prediction by showing:
# the ground truth, original model output, and final predictions generated using softmax and argmax of model output 
def display_prediction_insight_masks(filenames, ds, dataset_type, UNet_model, subset_results_subfolder, device, out_of_distribution = False): 
    
    #define subfolder to save images:
    img_insight_folder = os.path.join(subset_results_subfolder, 'ModelPredictionInsightImages')
    if not os.path.exists(str(img_insight_folder)):
        os.mkdir(str(img_insight_folder))
    
    #iterate through images in subset (validation or test) files:
    for index in range(0,len(filenames)):
        with torch.no_grad():
            #select one image to evaluate and visualize the model output
            input = ds[index]["image"].unsqueeze(0).to(device)
            filename = filenames[index]
            output = UNet_model(input)
            plt.figure() 
            plot_idx = 0
            cols = ["background", "bladder", "anterior cervix", "posterior cervix", "cervical canal", "bladder"]
            rows = ['ground truth', 'softmax output', 'predicted mask']
            fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 8))
            font = {'size': 10}
            plt.rc('font', **font)
            for ax, col in zip(axes[0], cols):
                ax.set_title(col)
            for ax, row in zip(axes[:,0], rows):
                ax.set_ylabel(row, rotation=90, size='large')
            #fill rows of figure:
            for j in range(3):
                if j==0: #ground truth
                    for i in range(5):
                        if 'label' in ds[index].keys():
                            gt_label = ds[index]["label"][i,:,:].detach().cpu()
                        else: #if no ground truth image exists
                            gt_label = torch.zeros((256,256), dtype=torch.bool)
                        axes[j, i].imshow(gt_label)
                        axes[j,i].tick_params(labelleft=False, labelbottom=False)  
                           
                elif j==1: #model output (before softmax and argmax are applied to generate prediction)
                    #visualize the 5 (class) channel model output corresponding to this image
                    for i in range(5):
                        axes[j, i].imshow(output[0,i,:,:].detach().cpu())
                        axes[j, i].tick_params(labelleft=False, labelbottom=False)
                        
                elif j==2: #prediction based on model (after softmax and argmax are applied)
                    for i in range(5):
                        softmax_output = torch.nn.functional.softmax(output, dim = 1) #softmax of model output
                        argmax_output = torch.argmax(softmax_output, dim=1)
                        #define unique_mask of arg_max() values, to be used as prediction in j==2
                        unique_mask = torch.zeros(output.shape)
                        for x in range(output.shape[2]):
                            for y in range(output.shape[3]):
                                idx = argmax_output[0,x,y]
                                unique_mask[0,idx,x,y] = 1 #reassign class values
                        axes[j, i].imshow(unique_mask[0,i,:,:].detach().cpu())
                        axes[j, i].tick_params(labelleft=False, labelbottom=False)
         
        #set prefix for file naming based on which dataset is being used     
        if out_of_distribution:
            img_insight_name = filename
        else:
            img_insight_name = dataset_type.capitalize() + '_' + filename  
        fig.savefig(os.path.join(img_insight_folder, img_insight_name))
        plt.close(fig)


# For the validation set, this function saves prediction images and metrics (dice score, hausdorff distance, jaccard index) to assess model performance:
def save_prediction_images_and_metrics(results_subfolder, UNet_model, ds, filenames, device, dice_metric, dataset_type, out_of_distribution = False):

    #define color list:
    color_list = ["yellow", "magenta", "cyan", "lime" ]
    
    #If data is within the original clear distribution:
    if not out_of_distribution:
        #make a validation or test folder: to save model prediction images and model metrics
        subset_folder_str = dataset_type.capitalize() + '_CLEAR'
        if not os.path.exists(subset_folder_str):
            os.mkdir(subset_folder_str)
        os.chdir(subset_folder_str)
    
        #make a validation/test folder: to save model prediction images and model metrics evaluated on validation/test set
        subset_results_subfolder = os.path.join(results_subfolder, subset_folder_str)
        if not os.path.exists(str(subset_results_subfolder)):
            os.mkdir(str(subset_results_subfolder))
    
    #If data is within additional dataset, out of the original distribution (ex: Bounce of LORI dataset)
    elif out_of_distribution:
        subset_predictions = dataset_type + '_predictions'
        subset_results_subfolder = os.path.join(results_subfolder,subset_predictions)
        if not os.path.exists(subset_results_subfolder):
            os.mkdir(subset_results_subfolder)
    #save images that provide some insight into how the model makes its prediction
    display_prediction_insight_masks(filenames, ds, dataset_type, UNet_model, subset_results_subfolder, device, out_of_distribution)
    

    #create list of prediction mask/image pairs:
    with torch.no_grad():
        gt_images_with_masks = []
        predicted_images_with_masks = []
        for index, image_pair in enumerate(ds):
            image =  (image_pair["image"]*255).to(torch.uint8)
            if 'label' in image_pair.keys():
                gt_label  = image_pair["label"].detach().cpu().to(torch.bool)
            else: #if no ground truth image exists
                gt_label = torch.zeros((5,256,256), dtype=torch.bool)
            #calculate predicted mask:
            input = ds[index]["image"].unsqueeze(0).to(device)
            output = UNet_model(input)
            unique_output = batched_post_trans(output)
            pred_label = unique_output[0].to(torch.bool)
            #if the image is 1 color channel, repeat this channel 3 times so it can be read as an RGB image:
            if image.shape[0] == 1:
                    image = image.repeat(3,1,1)
            #draw ground truth and prediction mask with transparency on top of original image, and append each to separate lists
            gt_segmentation_mask = torchvision.utils.draw_segmentation_masks(image,masks = gt_label[1:], alpha = 0.6, colors = color_list)
            gt_images_with_masks.append(gt_segmentation_mask)
            pred_segmentation_mask = torchvision.utils.draw_segmentation_masks(image,masks = pred_label[1:], alpha = 0.6, colors = color_list)
            predicted_images_with_masks.append(pred_segmentation_mask)
            #add a batch dimension, needed to calculate dice metric
            pred_label = torch.unsqueeze(pred_label, dim = 0)
            gt_label = torch.unsqueeze(gt_label, dim = 0)
            
            #compute dice score for a single image
            single_label_dice = dice_metric(y_pred=pred_label, y=gt_label)
            #compute hausdorff distance for a single image
            single_label_HD = monai.metrics.compute_hausdorff_distance(y_pred = pred_label, y = gt_label, include_background = True)
            #compute jaccard index for a single image           
            single_label_jaccard = sklearn.metrics.jaccard_score(y_true = gt_label.flatten(2,3).squeeze(0).swapaxes(0,1), y_pred = pred_label.flatten(2,3).squeeze(0).swapaxes(0,1), average = None)           
            single_label_jaccard = torch.from_numpy(single_label_jaccard).unsqueeze(0)

            #calculate average dice metric, hausdorff distance, and jaccard index for each image:
            single_label_dice_avg = torch.mean(single_label_dice).unsqueeze(0).unsqueeze(0)
            single_label_HD_avg = torch.mean(single_label_HD).unsqueeze(0).unsqueeze(0)
            single_label_jaccard_avg = torch.mean(single_label_jaccard).unsqueeze(0).unsqueeze(0)
            #generate tensor of dice score, hausdorff distance, and jaccard index for all images in validation/test set
            if index==0:
                all_data_dice = single_label_dice
                all_data_HD = single_label_HD
                all_data_jaccard = single_label_jaccard
                all_data_dice_avg = single_label_dice_avg
                all_data_HD_avg = single_label_HD_avg
                all_data_Jaccard_avg = single_label_jaccard_avg
                print("{} all_data_dice_avg = {}".format(dataset_type, all_data_dice_avg.shape))
                print("{} all_data_HD_avg  = {}".format(dataset_type, all_data_HD_avg.shape))
                print("{} all_data_Jaccard_avg = {}".format(dataset_type, all_data_Jaccard_avg.shape))
            else:
                all_data_dice = torch.cat((all_data_dice, single_label_dice), dim = 0)
                all_data_HD = torch.cat((all_data_HD, single_label_HD), dim=0)
                all_data_jaccard = torch.cat((all_data_jaccard, single_label_jaccard), dim=0)
                #concatenate average dice metric, hausdorff distance, and jaccard index along image dimension
                all_data_dice_avg = torch.cat((all_data_dice_avg, single_label_dice_avg) , dim = 0)
                all_data_HD_avg = torch.cat((all_data_HD_avg, single_label_HD_avg) , dim = 0)
                all_data_Jaccard_avg = torch.cat((all_data_Jaccard_avg, single_label_jaccard_avg) , dim = 0)
           
            dice_metric.reset()
    
    #save side-by-side image comparing ground truth segmentation to predicted segmentation
    display_gt_vs_prediction_mask(gt_images_with_masks, predicted_images_with_masks, subset_results_subfolder, filenames, dataset_type, out_of_distribution)

    #calculate class-specific average and standard deviation for each metric, across all images in validation/test set
    # each of the 6 variables below should be of dimension 1-by-#classes, in this case [1,5]
    across_data_dice = torch.nanmean(all_data_dice, dim=0)
    across_data_dice_std = torch.std(all_data_dice, dim=0)
    across_data_HD = torch.nanmean(all_data_HD, dim=0)
    across_data_HD_std = torch.std(all_data_HD, dim=0)
    across_data_jaccard = torch.nanmean(all_data_jaccard, dim=0)
    across_data_jaccard_std = torch.std(all_data_jaccard, dim=0)
    print("{} across_data_dice is: {} +/- {}".format(dataset_type, across_data_dice, across_data_dice_std))
    print("{} across_data_HD is: {} +/- {}".format(dataset_type, across_data_HD, across_data_HD_std))
    print("{} across_data_jaccard is: {} +/- {}".format(dataset_type, across_data_jaccard, across_data_jaccard_std))
    
    #calculate average and standard deviation (of class-average dice metric, hausdorff distance, and jaccard index) across all images in subset (validation/test):
    across_data_avg_dice = torch.nanmean(all_data_dice_avg, dim = 0)
    across_data_avg_dice_std = torch.std(all_data_dice_avg, dim = 0)
    across_data_avg_HD = torch.nanmean(all_data_HD_avg, dim = 0)
    across_data_avg_HD_std = torch.std(all_data_HD_avg, dim = 0)
    across_data_avg_jaccard = torch.nanmean(all_data_Jaccard_avg, dim = 0)
    across_data_avg_jaccard_std = torch.std(all_data_Jaccard_avg, dim = 0)
    
    # SAVE VALIDATION/TEST METRICS TO CSV FILE:
    metric_names = ['Dice', 'Dice_std', 'HD', 'HD_std', 'Jaccard', 'Jaccard_std']
    
    average_metrics = [across_data_avg_dice.item(), across_data_avg_dice_std.item(),
                       across_data_avg_HD.item(), across_data_avg_HD_std.item(),
                       across_data_avg_jaccard.item(), across_data_avg_jaccard_std.item()]
    
    background_metrics = [across_data_dice[0].item(), across_data_dice_std[0].item(),
                          across_data_HD[0].item(), across_data_HD_std[0].item(),
                          across_data_jaccard[0].item(), across_data_jaccard_std[0].item()]
    
    bladder_metrics = [across_data_dice[1].item(), across_data_dice_std[1].item(),
                       across_data_HD[1].item(), across_data_HD_std[1].item(),
                       across_data_jaccard[1].item(), across_data_jaccard_std[1].item()]
    
    anterior_metrics = [across_data_dice[2].item(), across_data_dice_std[2].item(),
                        across_data_HD[2].item(), across_data_HD_std[2].item(),
                        across_data_jaccard[2].item(), across_data_jaccard_std[2].item()]
    
    posterior_metrics = [across_data_dice[3].item(), across_data_dice_std[3].item(),
                         across_data_HD[3].item(), across_data_HD_std[3].item(),
                         across_data_jaccard[3].item(), across_data_jaccard_std[3].item()]
    
    canal_metrics = [across_data_dice[4].item(), across_data_dice_std[4].item(),
                     across_data_HD[4].item(), across_data_HD_std[4].item(),
                     across_data_jaccard[4].item(), across_data_jaccard_std[4].item()]
    
    metric_var_filename = dataset_type.capitalize() + '_Metric_variables.csv'
    df = pd.DataFrame({'Metric Name': metric_names, 'avg': average_metrics, 'background': background_metrics, 'bladder': bladder_metrics, 'anterior': anterior_metrics, 'posterior': posterior_metrics, 'canal': canal_metrics})
    df.to_csv(os.path.join(subset_results_subfolder, metric_var_filename), index=False)

    #save detailed dice scores and name of corresponding image (for each image in the validation set) to csv
    #dice Variables:
    dice_avg_col = []
    dice_background_col = []
    dice_bladder_col = []
    dice_anterior_col = []
    dice_posterior_col = []
    dice_canal_col = []
    #Hausdorff Distance Variables:
    HD_avg_col = []
    HD_background_col = []
    HD_bladder_col = []
    HD_anterior_col = []
    HD_posterior_col = []
    HD_canal_col = []
    #Jaccard Index Variables
    Jaccard_avg_col = [] #initialize as empty and then fill
    Jaccard_background_col = []
    Jaccard_bladder_col = []
    Jaccard_anterior_col = []
    Jaccard_posterior_col = []
    Jaccard_canal_col = []
    
    for i, filename in enumerate(filenames):
        #dice-----------------------------------------------------------------
        #take average of all classes except background
        dice_avg_col.append(torch.nanmean(all_data_dice[i][1:-1]).item())
        #take class-specific dice scores
        dice_background_col.append(all_data_dice[i][0].item())
        dice_bladder_col.append(all_data_dice[i][1].item())
        dice_anterior_col.append(all_data_dice[i][2].item())
        dice_posterior_col.append(all_data_dice[i][3].item())
        dice_canal_col.append(all_data_dice[i][4].item())
        #Hausdorff------------------------------------------------------------
        #take average of all classes except background
        HD_avg_col.append(torch.nanmean(all_data_HD[i][1:-1]).item())
        #take class-specific dice scores
        HD_background_col.append(all_data_HD[i][0].item())
        HD_bladder_col.append(all_data_HD[i][1].item())
        HD_anterior_col.append(all_data_HD[i][2].item())
        HD_posterior_col.append(all_data_HD[i][3].item())
        HD_canal_col.append(all_data_HD[i][4].item())
        #Jaccard---------------------------------------------------------------
        #take average of all classes except background
        Jaccard_avg_col.append(torch.nanmean(all_data_jaccard[i][1:-1]).item())
        #take class-specific dice scores
        Jaccard_background_col.append(all_data_jaccard[i][0].item())
        Jaccard_bladder_col.append(all_data_jaccard[i][1].item())
        Jaccard_anterior_col.append(all_data_jaccard[i][2].item())
        Jaccard_posterior_col.append(all_data_jaccard[i][3].item())
        Jaccard_canal_col.append(all_data_jaccard[i][4].item())
        
    #assign dataframe column names and variables, save to csv
    dice_df_each_image = pd.DataFrame({'Filename': filenames, 'avg_dice_noBackground': dice_avg_col,
                                       'background_dice': dice_background_col,'bladder_dice': dice_bladder_col,
                                       'anterior_dice': dice_anterior_col,'posterior_dice': dice_posterior_col,
                                       'canal_dice': dice_canal_col})
    all_im_dice_filename = dataset_type.capitalize() + '_AllImages_DiceScores.csv'
    dice_df_each_image.to_csv(os.path.join(subset_results_subfolder, all_im_dice_filename), index=False)

    HD_df_each_image = pd.DataFrame({'Filename': filenames, 'avg_HD_noBackground': HD_avg_col,
                                       'background_HD': HD_background_col,'bladder_HD': HD_bladder_col,
                                       'anterior_HD': HD_anterior_col,'posterior_HD': HD_posterior_col,
                                       'canal_HD': HD_canal_col})
    all_im_HD_filename = dataset_type.capitalize() + '_AllImages_HDScores.csv'
    HD_df_each_image.to_csv(os.path.join(subset_results_subfolder, all_im_HD_filename), index=False)

    Jaccard_df_each_image = pd.DataFrame({'Filename': filenames, 'avg_jaccard_noBackground': Jaccard_avg_col,
                                       'background_jaccard': Jaccard_background_col,'bladder_jaccard': Jaccard_bladder_col,
                                       'anterior_jaccard': Jaccard_anterior_col,'posterior_jaccard': Jaccard_posterior_col,
                                       'canal_jaccard': Jaccard_canal_col})
    
    all_im_Jaccard_filename = dataset_type.capitalize() + '_AllImages_JaccardScores.csv'
    Jaccard_df_each_image.to_csv(os.path.join(subset_results_subfolder, all_im_Jaccard_filename), index=False)

    return across_data_dice
