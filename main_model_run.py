# ---------------------------------------------------------------------------------------------------------------------------- #
#  This is the primary script used to:
#  1. load the images and label from source (folder structure or Labelbox directly)
#  2. Convert data to proper format needed for model input
#  3. Initialize model architecture
#  4. Train model, saving training/validation metrics
#  5. Evaluate model performance on test and validation set, save plots and prediction images
# ---------------------------------------------------------------------------------------------------------------------------- #

# Load packages:
import sys
import json
import torch
import numpy as np
import os
import pandas as pd
from glob import glob #for model predictions and evaluations on new datasets
import monai #for training UNet model
from monai.metrics import DiceMetric
from monai.utils import set_determinism

# Import custom python modules/functions:
import AMF_code
import my_transforms
import pre_processing
import model_utils
import save_model_metrics
import Inpaint_func
import Labelbox_save2dir


def main_model_run(device_number):

    # ---------------------------------------------------------------------------------------------------------------------------- #
    #  Set hyperparameters and model inputs
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Define json file to read input parameters from:
    model_args = 'model_arg' + str(device_number) + '.json'
    print(model_args)
    # Read hyperparameters from json input
    model_arguments_file = os.path.join('/home/obsegment/code/mutliclass_segmentation_TVUS', model_args)
    f = open(model_arguments_file)
    model_arguments = json.load(f)
    # Define python variables from json file variables:
    model_type =  model_arguments['model_type'] #UNet for example
    optim = model_arguments['optim']
    lr_stepsize = model_arguments['lr_stepsize']
    lr_gamma = model_arguments['lr_gamma']
    dropout = model_arguments['dropout']
    n_epochs = model_arguments['n_epochs']
    batch_size = model_arguments['batch_size']
    augmentation_flag = model_arguments['augmentation_flag']
    aug_type = model_arguments['aug_type']
    use_AMF_images = model_arguments['use_AMF_images']
    use_Inpaint_images = model_arguments['use_Inpaint_images']
    num_res_units = model_arguments['num_res_units'] #define number of residual units - default to 2
    es_patience = model_arguments['es_patience'] #early stopping, patience
    
    # Other hyperparameters:
    training_channels = 1 #channels to train model with
    n_classes=5 #number of classes

    # Configuration parameters:
    load_images_from_labelbox = False #whether to load images from labelbox
    apply_AMF_flag = False #whether to apply inpainting to the loaded images
    apply_inpaint_flag = True #
    rerun_model = True #change to true if you want to evaluate the model outputs but have already trained the model
    model_folder = None #change to directory of previously-trained model, if you set rerun_model = True because you want to skip training
    model_folder = '20220724_UNet_ch1Res4_Inpaint_optimAdam_lr0p001_dropoutp2_batchsz16_epoch5_patience5_BestValDice_augRcGsGnRtCrRz_numAug1026'
    cross_validation = False

    # Manually set the folder we want to reference which has images and associated labels - If we pull data from Labelbox more than once, this allows us to set which folder we reference as our dataset:
    date_labelbox_pull = '20220625' #date of image folder we want to reference - if not pulling data from labelbox in this run
    CLEAR_root_path = '/home/obsegment/code/ResearchDataset/CLEAR'
    raw_img_path = os.path.join(CLEAR_root_path, 'data')
    results_folder = os.path.join(CLEAR_root_path, 'Results')
    os.chdir(CLEAR_root_path)
    
    #set determinism for reproducability:
    set_determinism(seed=0) #this will maintain that the initial weights of the model are the same, the train/test/split will also be consistent
    
    
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # STEP 0:
    # ---------------------------------------------------------------------------------------------------------------------------- #
    print('Step 0 is running')
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # If the images are not already pre-loaded on the machine, pull them from labelbox. Apply AMF or inpainting filtering as needed,
    # to remove calipers from US images if this has not already been done.
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # If you want to import updated images/labels from labelbox, run the following code chunk (by setting load_images_from_labelbox to True)
    # ---------------------------------------------------------------------------------------------------------------------------- #

    save_imglabel_path = None # initiaize path where images/labels will be saved
    if load_images_from_labelbox: #if images will be loaded from Labelbox, use Labelbox.save2dir() to define path
        save_imglabel_path = Labelbox_save2dir.LoadAndSaveImages(raw_img_path)
    
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Run adaptive median filtering: save images to the same folder
    # ---------------------------------------------------------------------------------------------------------------------------- #
    
    #define path to save adaptive median filtered images:
    if load_images_from_labelbox:
        start_AMF_path = save_imglabel_path
    else:
        start_AMF_path = os.path.join(raw_img_path, date_labelbox_pull)
   

    #apply adaptive median filtering, save filtered images to start_AMF_path:
    if apply_AMF_flag:
        AMF_code.apply_AMF(start_AMF_path)

    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Run inpainting: save images to the same folder
    # ---------------------------------------------------------------------------------------------------------------------------- #
    
    #define path to save inpainted images:
    if load_images_from_labelbox:
        start_inpaint_path = save_imglabel_path
    else:
        start_inpaint_path = os.path.join(raw_img_path, date_labelbox_pull)

    #apply inpainting filtering, save filtered images to start_inpaint_path:
    if apply_inpaint_flag:
        img_prefix = '*_im*'
        image_paths = sorted(glob(os.path.join(start_inpaint_path, img_prefix))) #list of image paths
        for img_path in image_paths:
            #only perform inpainting on images which have not already been filtered (no AMF or Inpaint in image path) and ignore labels (lab in image path):
            if (not 'lab' in img_path) and (not 'AMF' in img_path) and (not 'Inpaint' in img_path):
                Inpaint_func.inpainting(img_path)
    
              
              
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # STEP 1:
    # ---------------------------------------------------------------------------------------------------------------------------- #
    print('Step 1 is running: load data from files for network training')
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Getting image/label paths, store them in dictionary
    # ---------------------------------------------------------------------------------------------------------------------------- #
    
    data_path = None #initialize path, used to access data (images and labels)
    if use_AMF_images: #adaptive median filtered images
        data_path = start_AMF_path
        img_prefix = '*AMF_*'
    elif use_Inpaint_images: #inpainted images
        data_path = start_inpaint_path
        img_prefix = '*Inpaint_*'
    else: #images which have not been filtered with AMF or Inpainting
        img_prefix = 'CLEAR_g[0-9]_im*' #look for this expression only in the beginning of the path
        if save_imglabel_path is not None: #this is true when we are loading images from labelbox
            data_path = save_imglabel_path
        else: #if path to save images and labels is still None:
            data_path = os.path.join(raw_img_path, date_labelbox_pull)
            print('No path provided for input images, check if this path is correct: ' + data_path)

    
    # Create list of image and segmentation label paths:
    images = sorted(glob(os.path.join(data_path, img_prefix))) #list of image paths
    segs = sorted(glob(os.path.join(data_path, "*lab[0-9][0-9][0-9]_majGT*"))) #list of mask paths
    
    # Verify that there are the same number of data in the image and label directory:
    print("there are {} images in the starting directory".format(len(images)))
    print("there are {} GT segmentations in the starting directory".format(len(segs)))
    
    # Fill image id list: this is a list of strings with format CLEAR_g#_im###
    image_id_list = []
    for i, img in enumerate(images):
        image_id = img.split('/')[-1].split('.')[0] #take everything before .png and after the folder path, save this as image_id
        #infer label_id based on image_id
        label_id = image_id.replace('im', 'lab') #label string format: CLEAR_g#_lab###
        if use_Inpaint_images:
            label_id = label_id.replace("Inpaint_", "") #rename label_id, because no filtering techniques were applied to labels
        
        # Only keep images with a corresponding label
        for s, seg in enumerate(segs):    
            if label_id in seg:
                image_id_list.append(img)
    
    # Verify that there are still the same number of items in the image id list and the segmentation label list
    print("length of image_id_list is {}".format(len(image_id_list)))
    print("length of segmentation list  is {}".format(len(segs)))
    
    # For each grade (6, 8, and 9), read images/labels from paths, convert to the datatype needed for training, and apply train-test-split:
    g6_train_files, g6_val_files, g6_test_files = pre_processing.prep_data_split(image_id_list, segs, 'g6', training_channels=training_channels)
    g8_train_files, g8_val_files, g8_test_files = pre_processing.prep_data_split(image_id_list, segs, 'g8', training_channels=training_channels)
    g9_train_files, g9_val_files, g9_test_files = pre_processing.prep_data_split(image_id_list, segs, 'g9', training_channels=training_channels)

    
    # Combine list of files for each group: train, validation and test:
    train_files = g6_train_files + g8_train_files + g9_train_files
    val_files = g6_val_files + g8_val_files + g9_val_files
    test_files = g6_test_files + g8_test_files + g9_test_files 

    # Extract file names from train/val/test list of dictionaries:
    train_filenames = []
    train_files_data = []
    for dict in train_files:
        train_filenames.append(dict["filename"])
        train_files_data.append({"image": dict["image"], "label": dict["label"]})
    val_filenames = []
    val_files_data = []
    for dict in val_files:
        val_filenames.append(dict["filename"])
        val_files_data.append({"image": dict["image"], "label": dict["label"]})
    test_filenames = []
    test_files_data = []
    for dict in test_files:
        test_filenames.append(dict["filename"])
        test_files_data.append({"image": dict["image"], "label": dict["label"]})
   
    
    # Review composition of train/val/test dataset:
    n_train_files = len(train_files)
    n_val_files = len(val_files)
    n_test_files = len(test_files)
    n_total_files = n_train_files + n_val_files + n_test_files
    print('n_train_files = ' + str(n_train_files))
    print('n_val_files = ' + str(n_val_files))
    print('n_test_files = ' + str(n_test_files))

    # Verify 70:20:10 split proportions:
    percent_train = int(np.round(n_train_files/n_total_files*100,0))
    percent_val = int(np.round(n_val_files/n_total_files*100,0))
    percent_test = int(np.round(n_test_files/n_total_files*100,0))
    print("The training set is {} percent of the dataset, the validation set is {} percent of the dataset and the test set is {} percent of the test set".format(percent_train,percent_val,percent_test))
    

    # ---------------------------------------------------------------------------------------------------------------------------- #
    #### STEP 2: ####
    # ---------------------------------------------------------------------------------------------------------------------------- #
    print('Step 2 is running: prepare batched data and apply augmentation')
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Apply augmentation and other transformations
    # ---------------------------------------------------------------------------------------------------------------------------- #
    #define dictionary to keep track of augmentations and input parameters:
    aug_params = {
        'RandAdjustContrastd' : {'prob':1.0, 'gamma':(0.5, 2.5), 'allow_missing_keys':False}, # Rc
        'GaussianSmoothd' : {'sigma' : 1}, # Gs
        'CustomGaussianNoised': {}, # Gn
        'Rotated_180': {'angle': np.pi, 'mode': ["bilinear", "nearest"]}, #Rt #invert image parameters
        'CenterSpatialCropd_200': {'roi_size': (200,200)}, # Cr
        'RandZoomd': {'prob': 1, 'min_zoom': 0.8, 'max_zoom': 0.9, 'mode': ["area", "nearest"]}, #Rz
    }

    aug_transformations = my_transforms.make_aug_transform(aug_params) #define augmentation transformations
    n_aug_transformations = len(list(aug_transformations)) #number of augmentation types applied
    
    # Prepare data for model training
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = pre_processing.load_data(train_files_data, val_files_data, test_files_data, batch_size, augmentation_flag, aug_transformations)
    
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Create result subfolder to store model outputs; save model parameters in CSV
    # ---------------------------------------------------------------------------------------------------------------------------- #
    
    if rerun_model == False:
        model_name, results_subfolder = save_model_metrics.create_model_run_folder(results_folder, aug_type, n_train_files, n_val_files, n_test_files, n_aug_transformations, num_res_units,
                                                                            lr_stepsize, lr_gamma, training_channels, optim, batch_size, dropout, n_epochs, use_AMF_images, use_Inpaint_images, model_type, es_patience)
    elif rerun_model == True:
        #use model_folder defined at top of code
        model_name = 'Model.pt'
        results_subfolder = os.path.join(results_folder, model_folder)
        
    '''# If we want to adjust the seed for cross-validation, this section can be used:
    if cross_validation == True:
        #rewrite over model name: to run cross validation
        model_folder =  model_arguments['model_name'] # from json file
        results_subfolder = os.path.join(results_folder, model_folder)
        if not os.path.exists(results_subfolder): #make results subfolder directory
            os.mkdir(results_subfolder)'''
    
    print('results_subfolder = ' + results_subfolder)
    
    # Save list of filenames for images allocated to each set (train/val/test) that may be accessed later:
    save_model_metrics.save_image_filenames(train_filenames, val_filenames, test_filenames, results_subfolder)

    # Save augmentation parameters to csv in specified path:
    my_transforms.save_aug_params(aug_params, aug_transformations, train_files, results_subfolder)
    

    # ---------------------------------------------------------------------------------------------------------------------------- #
    #### STEP 3: ####
    # ---------------------------------------------------------------------------------------------------------------------------- #
    print('Step 3 is running: define model architecture')
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Model definition and metrics
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # define model, loss and optimizer
    device = torch.cuda.set_device(int(device_number))
    
    if model_type == 'UNet':
        model = monai.networks.nets.UNet(
            dimensions=2, #2d image
            in_channels=training_channels, # color channels
            out_channels=5, #0 1 2 3 classes
            channels=(16, 32, 64, 128, 256), #convolution channels
            strides=(2, 2, 2, 2),
            num_res_units=num_res_units, #2 default
            dropout=dropout #add dropout
        ).to(device)

    elif model_type == 'AttentionUNet':
        model = monai.networks.nets.AttentionUnet(
            spatial_dims=2, #2d image
            in_channels=training_channels, # color channels
            out_channels=5, #0 1 2 3 classes
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            kernel_size = 3, #default value
            up_kernel_size = 3, #default value
            dropout=dropout #add dropout
        ).to(device)
        
    elif model_type == 'UNETR':
        model = monai.networks.nets.UNETR(
            in_channels=training_channels, # color channels
            out_channels=5, #0 1 2 3 classes
            img_size = (256,256), # default image size, corresponding to spatial_size in my_transforms.py
            feature_size = 16, #default value
            hidden_size = 768, #default value
            mlp_dim = 3072, #default value
            num_heads = 12, #default value
            pos_embed = 'conv', #default value
            norm_name = 'instance', #default value
            conv_block = True, #default value
            res_block = True, #default value
            dropout_rate =dropout, #add dropout
            spatial_dims =2 #2D image
        ).to(device)
        
    elif model_type == 'SegResNet':
        model = monai.networks.nets.SegResNet(
            #default init filters
            spatial_dims=2, #2d image
            in_channels=training_channels, # color channels
            out_channels=5, #0 1 2 3 classes
            dropout_prob=dropout #add dropout
            #default activation
            #default convolution, and upsample mode
        ).to(device)
        
    elif model_type == 'SegResNetVAE':
        model = monai.networks.nets.SegResNetVAE(
            input_image_size = (256,256), # default image size, corresponding to spatial_size in my_transforms.py
            vae_estimate_std = False,
            vae_default_std = 0.3,
            vae_nz = 256,
            spatial_dims = 3,
            init_filters = 8,
            in_channels = 1,
            out_channels = 2,
            dropout_prob = None,
            use_conv_final = True,
            blocks_down = (1,2,2,4),
            blocks_up = (1,1,1)
            #default upsample_mode
        ).to(device)
        
    else:
        print("This model_type hasn't been defined yet")
        
    print("model type = {}".format(model_type))

    
    #define metrics and loss functions used for training:
    loss_function = monai.losses.DiceLoss(include_background = True, to_onehot_y = False, softmax=True) #loss function fed into model training
    dice_metric = DiceMetric(include_background=True, reduction="mean") #dice metric to evaluate model performance during training

    # Define optimizer parameters:
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr_gamma) #lr_gamma defined in model_arg#.json, default lr_gamma for adam optimizer = 1e-3
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr_gamma) #lr_gamma defined in model_arg#.json, default lf_gamma for adam optimizer =0.1
    
    #Optional learning rate scheduler, not currently implemented:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=lr_gamma)    
    
    
    # ---------------------------------------------------------------------------------------------------------------------------- #
    #### STEP 4: ####
    # ---------------------------------------------------------------------------------------------------------------------------- #
    print('Step 4 is running: model training')
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Model Training
    # ---------------------------------------------------------------------------------------------------------------------------- #
    #If this is the first time we are training a model, then enter into train_model() function
    if rerun_model == False:
        best_epoch, train_loss, val_loss, train_meanDICE, val_meanDICE, train_dice_df, val_dice_df, best_ClassAvg_train_DICEstd, best_ClassAvg_val_DICEstd = model_utils.train_model(model, n_epochs, optimizer, loss_function, dice_metric, device, n_classes, 
                                                                                    train_loader, val_loader, results_subfolder, model_name, batch_size, es_patience)
        
    #If this model has already been trained, and we only want to re-plot, re-visualize predictions, or re-calculate performance metrics, skip train_model() function and reference results_subfolder:
    elif rerun_model == True:
        # Access model performance data from saved csv files within model folder:
        best_epoch_path = os.path.join(results_subfolder, 'ModelConvergenceTimeMetrics.csv')
        best_epoch_df = pd.read_csv(best_epoch_path)
        best_epoch = best_epoch_df.loc[0].at['best_metric_epoch']
        
        train_var_csv_path = os.path.join(results_subfolder,'TrainVal_LossDice_variables.csv')
        training_variables_df = pd.read_csv(train_var_csv_path)
        train_loss = training_variables_df['train_loss'].tolist()
        val_loss = training_variables_df['val_loss'].tolist()
        train_meanDICE = training_variables_df['train_meanDICE'].tolist()
        val_meanDICE = training_variables_df['val_meanDICE'].tolist()
        
        #read csv with saved dice and loss metrics from results_subfolder: class-specific values
        val_dice_df = pd.read_csv(os.path.join(results_subfolder,'ValDice_variables.csv'))
        train_dice_df = pd.read_csv(os.path.join(results_subfolder,'TrainDice_variables.csv'))
            
        
    # ---------------------------------------------------------------------------------------------------------------------------- #
    #### STEP 5: ####
    # ---------------------------------------------------------------------------------------------------------------------------- #
    print('Step 5 is running: save model metrics and performance outputs')
    # ---------------------------------------------------------------------------------------------------------------------------- #
    # Saving model metrics and performance outputs
    # ---------------------------------------------------------------------------------------------------------------------------- #
    
    print("Saving loss and dice plots")
    save_model_metrics.save_loss_dice_plots(train_loss, val_loss, train_meanDICE, val_meanDICE, results_subfolder, best_epoch)
    
    print("Saving class-specific dice plots")
    save_model_metrics.save_class_dice_plots(val_meanDICE, train_dice_df, val_dice_df, results_subfolder, best_epoch)
    
    print("Saving predictions and metrics on validation set")
    val_across_data_dice = save_model_metrics.save_prediction_images_and_metrics(results_subfolder, model, val_ds, val_filenames, device, dice_metric, dataset_type='val')
    #save_model_metrics.save_val_prediction_images_and_metrics(results_subfolder, model, val_files, val_ds, device, dice_metric, val_filenames)
    #                                                       (results_subfolder, UNet_model, val_files, val_ds, device, dice_metric, val_filenames)
    
    print("Saving predictions and metrics on test set")
    test_across_data_dice = save_model_metrics.save_prediction_images_and_metrics(results_subfolder, model, test_ds, test_filenames, device, dice_metric, dataset_type='test')

    #test_across_data_dice, test_across_data_avg_dice = save_model_metrics.save_CLEAR_test_prediction_images(results_subfolder, model, test_files, test_ds, device, dice_metric, test_filenames)
    
    # Quick read-out of model performance on test set, see saved csv files for more specific metrics:
    print('test dice metric for each class = {}'.format(test_across_data_dice))
    print('test dice metric (average across classes) = {}'.format(torch.mean(test_across_data_dice, dim=0)))
    
    
if __name__ == '__main__':
    # Map command line arguments to function arguments.
    main_model_run(*sys.argv[1:])
    