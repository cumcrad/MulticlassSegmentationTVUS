# Load packages:
import torch
from monai.data import decollate_batch 
import os
from datetime import timedelta
import time
import numpy as np
import pandas as pd
from pytorch_utils import EarlyStopping

# Convert model output prediction to unique class values for each pixel:
def post_trans(model_output): #post-training transformation
    #take softmax across the channel/class dimension
    soft_out = torch.nn.functional.softmax(model_output, dim=0)
    #find argument (class assignment) with the highest predicted value
    class_idx = torch.argmax(soft_out, dim=0)
    #initialize empty mask to reassign with one-hot encoded class values
    unique_mask = torch.zeros(model_output.shape)
    for x in range(model_output.shape[1]):
            for y in range(model_output.shape[2]):
                idx = class_idx[x,y]
                unique_mask[idx,x,y] = 1 #reassign class value

    return unique_mask

# Convert model output prediction to unique class values for each pixel:
# this version handles batched data
def batched_post_trans(output): ##takes batched data
    #take softmax across the channel/class dimension
    soft_out = torch.nn.functional.softmax(output, dim=1)
    #find argument (class assignment) with the highest predicted value
    class_idx = torch.argmax(soft_out, dim=1)
    #initialize empty mask to reassign with one-hot encoded class values
    unique_mask = torch.zeros(output.shape)

    for b in range(output.shape[0]): #batch dimension
        for x in range(output.shape[2]): #spatial dimension
            for y in range(output.shape[3]):
                idx = class_idx[b,x,y]
                unique_mask[b,idx,x,y] = 1 #reassign class value

    return unique_mask


# Model training function:
def train_model(model, n_epochs, optimizer, loss_function, dice_metric, device, n_classes, train_loader, val_loader, results_subfolder, model_name, batch_size, es_patience, debug=False):

    #initialize variable to be filled
    start_time = time.perf_counter()
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    val_epoch_loss_values = list()
    val_meanDICE = list()
    val_meanDICE_std = list()
    train_meanDICE = list()
    train_meanDICE_std = list()
    train_dice_metric = np.empty((n_epochs, n_classes)) #dice metric per class (5 columns), 1 row per epoch
    train_dice_metric[:] = np.NaN
    val_dice_metric = np.empty((n_epochs, n_classes))
    val_dice_metric[:] = np.NaN
    
    num_train_batches = len(train_loader)
    if debug:
        print("there are {} train  batches".format(num_train_batches))
        print("there are {} elements in train dataset".format(len(train_loader.dataset)))
    
    num_val_batches = len(val_loader)
    if debug:
        print("there are {} validation batches".format(num_val_batches))
        print("there are {} elements in validation dataset".format(len(val_loader.dataset)))
    
    train_allImages_singleEpoch_dice_metric = np.empty((num_train_batches, batch_size, n_classes))
    train_allImages_singleEpoch_dice_metric[:] = np.NaN
    val_allImages_singleEpoch_dice_metric = np.empty((num_val_batches, batch_size, n_classes))
    val_allImages_singleEpoch_dice_metric[:] = np.NaN
    
    ClassAvg_train_allImages_allEpochs_dice_metric = np.empty((n_epochs, 2, n_classes))
    ClassAvg_train_allImages_allEpochs_dice_metric[:] = np.NaN
    ClassAvg_val_allImages_allEpochs_dice_metric = np.empty((n_epochs, 2, n_classes))
    ClassAvg_val_allImages_allEpochs_dice_metric[:] = np.NaN
    
    loss_difference_list = []

    #define early stopping
    patience = es_patience
    early_stopping = EarlyStopping(patience=patience, verbose=True, results_subfolder=results_subfolder)

    #start training loop:
    for epoch in range(n_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epochs}")
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            raw_inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            inputs = raw_inputs
            optimizer.zero_grad()
            outputs = model(inputs)
            #calculate loss:
            loss = loss_function(outputs, labels) 
            loss.backward()
            optimizer.step()

            #add loss for each batch to epoch loss
            epoch_loss += loss.item()

            #calculate training dice score:
            train_outputs = [post_trans(i) for i in decollate_batch(outputs)] 

            #calculate dice metric across batch (all images in a single batch) for each class:
            train_value = dice_metric(y_pred=train_outputs, y=labels) #train_value will return a tensor of size [batch_size, n_classes]
            if debug:
                print("train value shape is {}".format(train_value.shape))
            
            #store dice metric for every training image in a single epoch:
            #this will be used to calculate avg and std of dice metric across images in train set
            train_allImages_singleEpoch_dice_metric[step -1, 0:train_value.shape[0] ,:] = train_value
          
            #reset dice metric
            dice_metric.reset()

        #calculate average loss for the current epoch:
        #for the current epoch: average loss per batch = (sum of loss over all batches)/ #batches
        epoch_loss /= step
        #append average epoch loss to list:
        epoch_loss_values.append(epoch_loss) #list of loss, 1 value per epoch
        
        #calculate avg dice and std across all images for the current epoch: (results in 1 value per class)
        ClassAvgDICE_train_allImages_singleEpoch_dice_metric = np.nanmean(train_allImages_singleEpoch_dice_metric,axis=(0,1))
        ClassAvgDICEstd_train_allImages_singleEpoch_dice_metric = np.nanstd(train_allImages_singleEpoch_dice_metric, axis=(0,1),dtype=np.float64)
        #fill avg and std deviation in single variable:
        ClassAvg_train_allImages_allEpochs_dice_metric[epoch,0,:] = ClassAvgDICE_train_allImages_singleEpoch_dice_metric
        ClassAvg_train_allImages_allEpochs_dice_metric[epoch,1,:] = ClassAvgDICEstd_train_allImages_singleEpoch_dice_metric
        
        #calculate avg dice metric of all images in epoch (across all classes except background) to monitor training:
        train_epoch_dice_metric_NoBackground = np.nanmean(train_allImages_singleEpoch_dice_metric[:,:,1:])
        train_epoch_dice_metric_std_NoBackground = np.nanstd(train_allImages_singleEpoch_dice_metric[:,:,1:])
         
        #for each epoch, keep track of average dice for each class: 5 columns, 1 row per epoch
        train_dice_metric[epoch, :] = ClassAvg_train_allImages_allEpochs_dice_metric[epoch,0,:]

        #print results during training:
        print(f"epoch {epoch + 1} average train loss: {epoch_loss:.4f}, average train meandice:{ train_epoch_dice_metric_NoBackground :.4f}") ##### train_epoch_meandice_metric

        #store train dice values (avg and std) in a list after each epoch
        train_meanDICE.append(train_epoch_dice_metric_NoBackground)
        train_meanDICE_std.append(train_epoch_dice_metric_std_NoBackground)

        #validation loop of model training:
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                counter= 0
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    counter+=1
                    val_outputs = model(val_images)

                    #calculate validation loss:
                    val_loss = loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item() #add loss from each batch in epoch

                    #add softmax and unique (argmax) function to output before calculating dice score
                    val_outputs = [post_trans(i).to(device) for i in decollate_batch(val_outputs)]
                    
                    #calculate dice metric on validation set (single batch of images)
                    val_value = dice_metric(y_pred=val_outputs, y=val_labels)
                    
                    #store dice metric for every validation image in a single epoch:
                    #this will be used to calculate avg and std of dice metric across images in validation set
                    val_allImages_singleEpoch_dice_metric[counter-1,  0:val_value.shape[0],:] = val_value
                                        
                    dice_metric.reset()

                #calculate average loss for the current epoch:
                #for the current epoch: average loss per batch = (sum of loss over all batches)/ #batches
                val_epoch_loss /= counter
                val_epoch_loss_values.append(val_epoch_loss)
                #calculate the difference between validation and training loss
                loss_difference = abs(val_epoch_loss-epoch_loss)
                loss_difference_list.append(loss_difference)
                
                #calculate avg dice and std across all images for the current epoch: (results in 1 value per class)
                ClassAvgDICE_val_allImages_singleEpoch_dice_metric = np.nanmean(val_allImages_singleEpoch_dice_metric,axis= (0,1)) #take average across class dimension
                ClassAvgDICEstd_val_allImages_singleEpoch_dice_metric = np.nanstd(val_allImages_singleEpoch_dice_metric,axis = (0,1), dtype=np.float64) #I need to find a way to handle Nan values when calculating std
                #fill avg and std deviation in single variable:
                ClassAvg_val_allImages_allEpochs_dice_metric[epoch,0,:] = ClassAvgDICE_val_allImages_singleEpoch_dice_metric
                ClassAvg_val_allImages_allEpochs_dice_metric[epoch,1,:] = ClassAvgDICEstd_val_allImages_singleEpoch_dice_metric
                
                #calculate avg dice metric of all images in epoch (across all classes except background) to monitor training:
                val_epoch_dice_metric_NoBackground = np.nanmean(val_allImages_singleEpoch_dice_metric[:,:,1:])
                val_epoch_dice_metric_std_NoBackground = np.nanstd(val_allImages_singleEpoch_dice_metric[:,:,1:])
    
                #for each epoch, keep track of average dice for each class: 5 columns, 1 row per epoch
                val_dice_metric[epoch, :] = ClassAvg_val_allImages_allEpochs_dice_metric[epoch,0,:]
                        
                #store train dice values in a list after each epoch
                val_meanDICE.append(val_epoch_dice_metric_NoBackground)
                val_meanDICE_std.append(val_epoch_dice_metric_std_NoBackground)
                
                #print results during validation:
                print(f"epoch {epoch + 1} average val loss: {val_epoch_loss:.4f}, val_meanDICE {val_epoch_dice_metric_NoBackground:.4f}")

                #save updated model state dictionary only if validation dice score improves (does not include background class)
                if val_epoch_dice_metric_NoBackground > best_metric:
                    best_metric = val_epoch_dice_metric_NoBackground
                    best_metric_epoch = epoch + 1
                    #save the updated model state dictionary:
                    torch.save(model.state_dict(), os.path.join(results_subfolder, "best_metric_model_segmentation2d_dict.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {}, current mean dice: {:.4f}, best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_epoch_dice_metric_NoBackground, best_metric, best_metric_epoch
                    )
                )
                
        #early stopping checks if validation loss has decreased
        #if validation loss has decreased, it will make a checkpoint of the current model
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    #print final results of model training  
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    #save dice scores and std values at best metric epoch, to csv file:
    best_ClassAvg_train_DICEstd = ClassAvg_train_allImages_allEpochs_dice_metric[best_metric_epoch-1,:,:]
    best_ClassAvg_val_DICEstd = ClassAvg_val_allImages_allEpochs_dice_metric[best_metric_epoch-1,:,:]
    best_epoch_dice_avg_std_df = pd.DataFrame({'dataset_type': ['Train', 'Validation'], 
                                  'avg_dice_noBackground': [train_meanDICE[best_metric_epoch-1], val_meanDICE[best_metric_epoch-1]] , 
                                  'background_dice': [best_ClassAvg_train_DICEstd[0,0], best_ClassAvg_val_DICEstd[0,0]], 
                                  'bladder_dice': [best_ClassAvg_train_DICEstd[0,1], best_ClassAvg_val_DICEstd[0,1]],
                                  'anterior_dice': [best_ClassAvg_train_DICEstd[0,2], best_ClassAvg_val_DICEstd[0,2]],
                                  'posterior_dice': [best_ClassAvg_train_DICEstd[0,3], best_ClassAvg_val_DICEstd[0,3]],
                                  'potential_space_dice': [best_ClassAvg_train_DICEstd[0,4], best_ClassAvg_val_DICEstd[0,4]],
                                  'std_dice_noBackground':[train_meanDICE_std[best_metric_epoch-1], val_meanDICE_std[best_metric_epoch-1]] ,
                                  'background_dice_std': [best_ClassAvg_train_DICEstd[1,0], best_ClassAvg_val_DICEstd[1,0]],
                                  'bladder_dice_std': [best_ClassAvg_train_DICEstd[1,1], best_ClassAvg_val_DICEstd[1,1]],
                                  'anterior_dice_std': [best_ClassAvg_train_DICEstd[1,2], best_ClassAvg_val_DICEstd[1,2]],
                                  'posterior_dice_std': [best_ClassAvg_train_DICEstd[1,3], best_ClassAvg_val_DICEstd[1,3]],
                                  'potential_space_dice_std': [best_ClassAvg_train_DICEstd[1,4], best_ClassAvg_val_DICEstd[1,4]]
                                  })
    best_epoch_dice_avg_std_filename = 'ModelConvergencePerformanceMetrics_Epoch' + str(best_metric_epoch) + '.csv'
    best_epoch_dice_avg_std_df.to_csv(os.path.join(results_subfolder,best_epoch_dice_avg_std_filename), index=False)

    #save final trained model
    torch.save(model, os.path.join(results_subfolder, model_name))

    #return elapsed time for model training, save to csv:
    end_time = time.perf_counter()
    train_time = str(timedelta(seconds = (end_time - start_time))) #in seconds
    model_training_speed = pd.DataFrame({'train_time': [train_time], 'best_avg_dice_metric': [best_metric], 'best_metric_epoch': [best_metric_epoch]})
    model_training_speed.to_csv(os.path.join(results_subfolder,'ModelConvergenceTimeMetrics.csv'), index=False)

    #save train/val dice and loss metrics for every epoch to csv file
    train_loss = epoch_loss_values
    val_loss = val_epoch_loss_values 
    epoch = range(1,len(train_loss)+1)
    metrics_to_save = {'epoch': epoch,'train_loss': train_loss, 'val_loss' : val_loss, 'train_meanDICE' : train_meanDICE, 'val_meanDICE' : val_meanDICE}
    df = pd.DataFrame(metrics_to_save)
    df.to_csv(os.path.join(results_subfolder,'TrainVal_LossDice_variables.csv'), index=False)

    #save class-specific dice metrics to csv so that we can plot and examine them individually later 
    train_dice_df = pd.DataFrame(train_dice_metric, columns=['background_dice', 'bladder_dice','anterior_dice', 'posterior_dice', 'potential_space_dice'])
    train_dice_df.index.name = 'epoch'
    train_dice_df.index +=1
    val_dice_df = pd.DataFrame(val_dice_metric, columns=['background_dice', 'bladder_dice','anterior_dice', 'posterior_dice', 'potential_space_dice'])
    val_dice_df.index.name = 'epoch'
    val_dice_df.index += 1
    train_dice_df.to_csv(os.path.join(results_subfolder,'TrainDice_variables.csv'), index=True)
    val_dice_df.to_csv(os.path.join(results_subfolder,'ValDice_variables.csv'), index=True)
    
    return best_metric_epoch, train_loss, val_loss, train_meanDICE, val_meanDICE, train_dice_df, val_dice_df, best_ClassAvg_train_DICEstd, best_ClassAvg_val_DICEstd


