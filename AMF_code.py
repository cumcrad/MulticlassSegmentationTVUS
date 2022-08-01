import os
import time
import datetime

# Apply AMF on images stored within img_folder_path (by executing the script AMF_func.py in parallel on each image):
def apply_AMF(img_folder_path):
    
    print("Applying AMF for images in {}".format(img_folder_path))
    start_time = time.perf_counter()

    os.environ['IMG_PATH'] = img_folder_path
    #change command_str depending on which folder you want to access:
    command_str = """find $IMG_PATH -maxdepth 1 -type f -iname "*.png" | parallel python /DataStor/obseg/ResearchDataset/UNetCode_separated/AMF_func.py -i {}"""
    os.system(command_str)

    end_time = time.perf_counter()
    run_time = str(datetime.timedelta(seconds=end_time - start_time)) 
    print('Elapsed time for AMF filtering of all images: = ' + run_time)
    
# Notes:----------------------------------------------------------------------------------------------
# To check current running process, run this command in a linux terminal: pgrep -au obsegment python
# To kill current processes, run the following command in the terminal: pkill -u obsegment python    
