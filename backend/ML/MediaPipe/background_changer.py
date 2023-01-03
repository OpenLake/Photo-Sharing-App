import cv2
import numpy as np
import mediapipe as mp

def matchImageSize(foreground , background):
    f_h, f_w = foreground.shape[:2]
    b_h, b_w = background.shape[:2]
    if b_h < f_h and b_w < f_w:
        return -1
    else:
        return cv2.resize(foreground, (b_w, b_h))
    

def backgroundChange(foreground_path, background_path):
    change_background_mp = mp.solutions.selfie_segmentation
    change_bg_segment = change_background_mp.SelfieSegmentation()
    sample_img = cv2.imread(foreground_path)
    bg_img = cv2.imread(background_path)
    # Resize background to same path as that of foreground
    sample_img = matchImageSize(sample_img , bg_img)
    if type(sample_img) == int:
        return "Please upload background image greater than foreground image"
    

    RGB_sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    result = change_bg_segment.process(RGB_sample_img)
    binary_mask = result.segmentation_mask > 0.9
    binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
    output_image = np.where(binary_mask_3, sample_img, 255)   
    
    output_image = np.where(binary_mask_3, sample_img, bg_img)     
    return output_image


cv2.imwrite("REsult.jpg", backgroundChange("/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/ML/MediaPipe/istockphoto-1332653761-170667a.jpg","/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/ML/MediaPipe/558996.jpg"))