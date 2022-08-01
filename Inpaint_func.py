# Load packages:
import numpy as np
import cv2

def sepia_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sepia_lower = np.array([np.round( 30 / 2), np.round(0.10 * 255), np.round(0.10 * 255)])
    sepia_upper = np.array([np.round( 45 / 2), np.round(0.60 * 255), np.round(0.90 * 255)])
    return cv2.inRange(hsv, sepia_lower, sepia_upper)

def bluetone_mask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	bluetone_lower = np.array([90,50,70])
	bluetone_upper = np.array([175,255,255])
	return cv2.inRange(hsv, bluetone_lower, bluetone_upper)

def inpainting(img_path):
    inpaint_method = cv2.INPAINT_TELEA
	#open ultrasound images from particular date of pull
    #read in only one image at a time:
    img = str(img_path) #this range of white shows image 17 better
    #index 22 is a sepia image which is going to be a problem!!!!!!!!
    image = cv2.imread(img)

    #convert to hsv, to determine thresholding values: - before checking what type of image it is 
    #converts to hsv color type
    hsv_image= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #In HSV space - set color ranges:
    lower_green = np.array([36,10,70]) #([36,50,70])
    upper_green = np.array([89,255,255]) 

    blue_lower = np.array([90,50,70])
    blue_upper = np.array([175,255,255]) #([128,255,255]) #there may need to be a higher first value? so I changed to 175

    cyan_lower = np.array([150,50,70])
    cyan_upper = np.array([175,255,255])
    
    #values for grayscale and blue tones:
    white_sensitivity = 0
    white_lower = np.array([0,0,255 - white_sensitivity]) #([0,0,231])
    white_upper = np.array([225,white_sensitivity,255]) #25
    yellow_lower = np.array([5,0,70]) #[25,50,70] --> what if we lower what is below 25?
    yellow_upper = np.array([35,255,255])

    #values for sepia tones only:
    sepia_yellow_lower = np.array([20,90,230])
    sepia_yellow_upper = np.array([35,255,255])
    sepia_white_sensitivity = 50
    sepia_white_lower = np.array([0,0,255 - sepia_white_sensitivity])
    sepia_white_upper = np.array([225,sepia_white_sensitivity,255])

    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper) #blue tones and grayscale
    sepia_yellow_mask = cv2.inRange(hsv_image, sepia_yellow_lower, sepia_yellow_upper) #sepia only
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper) #blue tones and grayscale
    sepia_white_mask = cv2.inRange(hsv_image, sepia_white_lower, sepia_white_upper) #sepia only
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper) #for grayscale and sepia
    cyan_mask = cv2.inRange(hsv_image, cyan_lower, cyan_upper) #for grayscale
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green) #for grayscale, bluetones and sepia

    #NOW CHECK WHETHER THE IMAGE IS GRAYSCALE, BLUETONE OR SEPIA BEFORE APPLYING INPAINTING:

    # identify percentage of image that is in sepia tones:
    sepia_image = sepia_mask(image)
    perc_sepia = cv2.countNonZero(sepia_image) / np.prod(image.shape[:2])

    #start with non-sepia images:
    if perc_sepia*100 < 1:
        #check if bluetone:
        bluetone_image = bluetone_mask(image)
        perc_bluetone = cv2.countNonZero(bluetone_image) / np.prod(image.shape[:2])
        #print("percent of bluetone per image is {}".format(np.round(perc_bluetone*100,2)))
        
        #if bluetone image
        if perc_bluetone*100 > 1: 
            ##  bluetone will take yellow, grenen, cyan and white filters
            combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, cyan_mask) #remove blue mask for image idk 9
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
    
        elif perc_bluetone*100 <= 1:
            ##  grayscale will take yellow, green, and blue filters - not white
            combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, blue_mask) #remove blue mask for image idk 9
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
    elif perc_sepia*100 >= 1:
        ##  sepia will take green, blue and white filters BUT NOT YELLOW
        combined_mask = cv2.bitwise_or(green_mask, blue_mask) #remove blue mask for image idk 9
        combined_mask = cv2.bitwise_or(combined_mask, sepia_white_mask)
        combined_mask = cv2.bitwise_or(combined_mask, sepia_yellow_mask)

    #now determine target		
    target = cv2.bitwise_and(image,image, mask=combined_mask)
    combined_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    #dilate the mask to make a larger area
    kernel = np.ones((3,3),np.uint8)
    dilated_combined_mask  = cv2.dilate(combined_target,kernel)
    #showimage(dilated_combined_mask, 'dilated_combined_mask')

    #inpaint:
    inpainted_image = cv2.inpaint(image, dilated_combined_mask, 1, inpaint_method)
    #showimage(inpainted_image, 'inpainted_image')

    input_img_path = img_path
    input_img_name = input_img_path.split('/')[-1]
    #inpaint_img_name = input_img_name.split('_')[0] + '_Inpaint_' + input_img_name.split('_')[1]  #this is the version before we update to format CLEAR_g6_im000.png
    #print("input image name is {}".format(input_img_name))
    inpaint_img_name = 'Inpaint_' + input_img_name
    

    output_folder = input_img_path.replace(input_img_name, '')
    output_filename = output_folder + inpaint_img_name
    cv2.imwrite(output_filename, inpainted_image)

