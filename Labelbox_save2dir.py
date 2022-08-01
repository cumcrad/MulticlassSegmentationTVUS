

from PIL import Image
import numpy as np
from datetime import date
import os
from labelbox import Client, Label
from labelbox import Client, OntologyBuilder
from labelbox.data.annotation_types import Geometry
import json

def LoadAndSaveImages(path):
     
    labelbox_credentials_file = open('/home/obsegment/code/credentials.json')
    
    labelbox_credentials = json.load(labelbox_credentials_file)
    
    ######################################################################## LOAD IMAGES FROM LABELBOX API ##################################################################
    API_KEY = labelbox_credentials['API_KEY']
    PROJECT_ID = labelbox_credentials['PROJECT_ID']
    # Only update this if you have an on-prem deployment
    ENDPOINT = "https://api.labelbox.com/graphql"
    
    client = Client(api_key=API_KEY, endpoint=ENDPOINT)
    
    project = client.get_project(PROJECT_ID)
    
    #source code: https://colab.research.google.com/github/Labelbox/labelbox-python/blob/develop/examples/basics/labels.ipynb#scrollTo=hired-tyler
    labels = project.label_generator()
    labels = labels.as_list()

    #Citation: https://github.com/Labelbox/labelbox-python/blob/develop/examples/label_export/images.ipynb

    hex_to_rgb = lambda hex_color: tuple(int(hex_color[i+1:i+3], 16) for i in (0, 2, 4))
    colors = {tool.name: hex_to_rgb(tool.color) for tool in OntologyBuilder.from_project(project).tools}
   

    ######################################################################## SEPARATE IMAGES AND LABELS ##################################################################
    # DEFINE DICTIONARY OF IMAGES AND LABELS
    #create list of objects:
    raw_images = []
    label_masks = []

    for label_it in labels:
        #access labels created by one particular user/expert:
        user_email = label_it.extra['Created By']
        if user_email == labelbox_credentials['user_email']:
            empty_canvas = np.zeros(label_it.data.value.shape)
            for annotation in label_it.annotations:
                if isinstance(annotation.value, Geometry):
                    image_np = label_it.data.value #read in each image
                    mask_np = annotation.value.draw(canvas = empty_canvas, color = colors[annotation.name], thickness = 1)
            label_masks.append(mask_np)
            raw_images.append(image_np)

    #access length of the lists:
    print("length of raw images list is {}".format(len(raw_images)))
    print("length of mask labels list is {}".format(len(label_masks)))

    json_dicts= [{"image": img, "label": label} for img, label in zip(raw_images,label_masks)] 
    
    ######################################################################## SAVE IMAGES TO LOCAL DISK ##################################################################
    #access today's date to save to subfolder path
    query_date = str(date.today()).replace('-', '')
    save_imglabel_path = os.path.join(path,query_date)

    if not os.path.exists(str(save_imglabel_path)):
        os.mkdir(str(save_imglabel_path))
    
    for idx, dict in enumerate(json_dicts):
        img = Image.fromarray(dict["image"].astype(np.uint8))
        CLEAR_img_name = 'CLEAR_im' + '{0:03}'.format(idx) + '.png'
        img.save(os.path.join(save_imglabel_path, CLEAR_img_name))
        label = Image.fromarray(dict["label"].astype(np.uint8))
        CLEAR_label_name = 'CLEAR_lab' + '{0:03}'.format(idx) + '.png'
        label.save(os.path.join(save_imglabel_path, CLEAR_label_name))

    return save_imglabel_path


CLEAR_root_path = '/home/obsegment/code/ResearchDataset/CLEAR'
raw_img_path = os.path.join(CLEAR_root_path, 'data')
LoadAndSaveImages(raw_img_path)

