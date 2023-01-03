import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from ML.DeepImageMattingPyTorch.config import device
from ML.DeepImageMattingPyTorch.data_gen import data_transforms
from ML.DeepImageMattingPyTorch.models import DIMModel

def change_background(background, img, trimap):

    model = DIMModel()
    model.load_state_dict(torch.load("ML/DeepImageMattingPyTorch/Matting.pth"))
    # checkpoint = torch.load("ML/DeepImageMattingPyTorch/BEST_checkpoint.tar")
    # model = checkpoint['model'].module
    model = model.to(device)
    model.eval()
    
    transformer = data_transforms['valid']
    # background = cv.imread(background_path)
    
    # img = cv.imread(foreground_path)
    img_copy = img.copy()
    h, w = img.shape[:2]
    print("Shape of foreground", img.shape)
    background = cv.resize(background, (w,h))
    height, width = background.shape[:2]
    print("Shape of background", background.shape)
    
    if height < h or width < w:
        raise Exception("Background Size smaller that foreground image")

    x = torch.zeros((1, 4, h, w), dtype=torch.float)
    image = img[..., ::-1]  # RGB
    image = transforms.ToPILImage()(image)
    image = transformer(image)
    x[0:, 0:3, :, :] = image

    # trimap = cv.imread(trimap_path, 0)
    print("Shape of trimap", trimap.shape)

    x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)
    # Move to GPU, if available
    x = x.type(torch.FloatTensor).to(device)

    with torch.no_grad():
        pred = model(x)
    
    pred = pred.cpu().numpy()
    print("Shape of predictions", pred.shape)
    pred = pred.reshape((h, w))
    pred[trimap == 0] = 0.0
    pred[trimap == 255] = 1.0

    # Computing Offsets
    new_width = (width - w)  // 2
    # new_height = (height - h)  // 2

    result= cv.copyMakeBorder(pred, height-h, 0, new_width, new_width, cv.BORDER_CONSTANT,value=(0,0,0))
    
    bg_r = background[:,:,2]
    bg_g = background[:,:,1]
    bg_b = background[:,:,0]

    img_copy = cv.copyMakeBorder(img_copy, height-h, 0, new_width, new_width, cv.BORDER_CONSTANT,value=(0,0,0))

    img_r = img_copy[:,:,2]
    img_g = img_copy[:,:,1]
    img_b = img_copy[:,:,0]
    
    # Now making predictions
    predictions = np.zeros((height, width,3))
    predictions[:,:,0] = result * img_b + (1-result)*bg_b
    predictions[:,:,1] = result * img_g + (1-result)*bg_g
    predictions[:,:,2] = result * img_r + (1-result)*bg_r

    return predictions


# change_background("/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/nick-fewings-Tpsaue6Y9OA-unsplash.jpg", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/foreground/pexels-photo-733872.webp", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/images/trimap.jpg", "/home/aman/Desktop/FossOverflow/Photo-Sharing-App/backend/ML/DeepImageMattingPyTorch/BEST_checkpoint.tar")