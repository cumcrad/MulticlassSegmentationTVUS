# Overview:

The main_model_run.py is the primary script to run all components, from training to evaluation. This function takes 1 input which is the GPU device number to be used. In order to run this script, you will need to provide input parameters in a separate json file, stored in the same location. The corresponding json file is accessed also based on the GPU device number (model_arg0.json corresponding to device 0, model_arg1.json to device 1, etc). For example, from command line, one can execute:
`{python3 main_model_run.py 0}`


# Loading data:

* Labelbox_save2dir.py loads the images and labels from one expert from LabelBox. This code references a 'credentials.json' file which should contain variables: user_name, project_id and api_key.

* AMF_code.py is used to execute adaptive median filtering in parallel on a directory of images. This script accesses functions within AMF_func.py, which calls utility functions in filter_utils.py.

* Inpaint_func.py applies inpainting to remove ultrasound calipers on an image.

* pre_processing.py contains preprocessing utility functions that:
    1. convert image/label paths to numpy arrays
    2. convert pixel color values to class values for label arrays
    3. prepare data (using 1&2) and performs a train-test-split of given grouping (ex: 'g6')
    4. load data as monai.data.DataLoader object

* my_transforms.py defines monai-compatible transforms to be applied to the train, validation and test datasets using monai.data.DataLoader. These transforms are called from pre_processing.py. These transforms include data augmentation as well as preparation of the original dataset.

# The model:

* model_utils.py contains the main model training routine and a post-processing function applied to generate one-hot encoded predictions from the model predictions.
* pytorch_utils.py contains the EarlyStopping class referenced within model_utils.py.

# Saving performance metrics:
* save_model_metrics.py calculate performance metrics (dice score, hausdorff distance, and jaccard index) and saves prediction images in model output folder. The functions within this script are called within main_model_run.py

# Evaluate model performance for pre-trained model:
* evaluate_model_performance.py loads a pretrained model from a specified folder path, generates prediction masks for a new set of images, and calculates performance metrics on those images if a ground truth value is available.

# Saving images and calculating inter-observer metrics:
* InterOperator_variability.ipynb calculates inter-operator dice scores, hausdorff distance, and jaccard index between different experts and the GT mask for each image. Results saved as individual image values and averaged across all images before being saved to csv stored within an "InterOperator_Metrics" folder within the results folder. Fliess kappa scores are also calculated between the 3 expert labels for each image: this Fliess kappa coefficient is also averaged across all images in the test set and saved to csv.

* This notebook also generates images with each experts' GT prediction mask overlayed with transparency on top of the underlying ultrasound image. These are stored within the "Image_Overlay" folder inside the "Interoperator_Metrics" folder.

* InterOperator_utils.py contains functions referenced within InterOperator_variability.ipynb.

# Statistical analysis:
* StatisticalAnalysis.ipynb perfoms a paired one-way ANOVA test and subsequent paired t-test with bonferonni corrections. Results are visualized within the notebook, but are not explicitly saved to disk.
