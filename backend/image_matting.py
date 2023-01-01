from ML.DeepImageMattingPyTorch.eval import change_background
from utils import download_weights
import cv2 as cv

def imageMatting(background_path, foreground_path, trimap_path):
    # Now we need to call the deep image matting algorithm 
    download_weights("ML/DeepImageMattingPyTorch/Matting.pth", "https://drive.google.com/file/d/14cCe_DA1F5ZEYgVPdfa6_HiIn7AfKQl5/view?usp=sharing")
    # foreground_image = cv.imread(os.path.join(foreground_folder_path, os.listdir(foreground_folder_path)[0]))
    # background_image = cv.imread(background_image_path)
    changed_img = change_background(cv.imread(background_path), cv.imread(foreground_path),cv.imread(trimap_path,0))

    cv.imwrite("FinalResult.jpg", changed_img)


# imageMatting("/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/nick-fewings-Tpsaue6Y9OA-unsplash.jpg", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/foreground/pexels-photo-733872.webp", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/trimap.jpg")
