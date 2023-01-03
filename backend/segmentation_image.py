from ML.yolov7segmentation.segment.predict import run
import cv2 as cv
import numpy as np
from utils import download_weights

def segmentation(foreground_folder_path):
    # Running the segmentation part from YOLOV7
    download_weights("ML/yolov7segmentation/yolov7-seg.pt", "https://drive.google.com/file/d/1mkM5XhvEF2vWe7tN9XQeszXBKwckehil/view?usp=sharing")
    img = cv.cvtColor(run(source=foreground_folder_path), cv.COLOR_BGR2GRAY)
    # Generating the trimap from the image
    img[img == 255] = 255
    img[img != 255] = 0
    kernel = np.ones((6, 6), np.uint8)
    erode_img = cv.erode(img, kernel,cv.BORDER_REFLECT,  iterations=2) 
    dilate_img = cv.dilate(img, kernel, iterations=2)
    unknown1 = cv.bitwise_xor(erode_img,img)
    unknown2 = cv.bitwise_xor(dilate_img, img)
    unknowns = cv.add(unknown1,unknown2)
    unknowns[unknowns==255]=127
    trimap = cv.add(erode_img,unknowns)
    cv.imwrite("images/trimap.jpg", trimap)   
   


# segmentation("/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/foreground")

# imageMatting("/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/nick-fewings-Tpsaue6Y9OA-unsplash.jpg", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/foreground/pexels-photo-733872.webp", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/trimap.jpg", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/ML/DeepImageMattingPyTorch/BEST_checkpoint.tar")