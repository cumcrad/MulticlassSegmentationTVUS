# Load packages:
import PIL.Image
import argparse
from filter_utils import AdaptiveMedianFilter, renoiseInBorder

# Helper function: Runs Adaptive Median Filter on a single image in provided img_path
def AMF_filtering(img_path, output_filename, initial_filter_size, AMF_Max_filter_size, decrements): 
    img = PIL.Image.open(str(img_path))
    img = img.convert('LA')
    width, height = img.size
    newimg = img.copy() #create copy to manipulate
    #apply adaptive median filtering:
    for i in range(AMF_Max_filter_size, 1, decrements): #default: increments of -3, stopping at 1
        img = newimg.copy()
        newimg = AdaptiveMedianFilter(i, width, height, img, newimg, init_filtersize=initial_filter_size) 
    #add noise back in to border of filtered image
    newimg = renoiseInBorder(1, width, height, newimg)
    color_img = newimg.convert('RGB')
    color_img.save(output_filename)
    
# ------------------------------------------------------------------------------
#  Main script: runs adaptive median filter for the image in input_img_path 
#  (this script is executed in parallel on a list of image paths by AMF_code.py)
# ------------------------------------------------------------------------------
# Tuneable parameters:
initial_filter_size = 3
AMF_Max_filter_size = 23
decrements = -3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "Directs the output to a name of your choice")
    args = parser.parse_args()

# Get path from input argument, clean the name and generate AMF image name:
input_img_path = args.input
input_img_name = input_img_path.split('/')[-1]
amf_img_name = 'AMF_' + input_img_name 

# Define full path for output AMF images:
output_folder = input_img_path.replace(input_img_name, '')
output_filename = output_folder + amf_img_name

#Apply AMF filtering on images only (i.e. not labels), if the image has not already been preprocessed (using AMF or Inpainting)
if (not 'lab' in input_img_name) and (not 'AMF' in input_img_name) and (not 'Inpaint' in input_img_name):
    AMF_filtering(input_img_path, output_filename, initial_filter_size, AMF_Max_filter_size, decrements)