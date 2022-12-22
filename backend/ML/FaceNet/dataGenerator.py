import numpy as np
import cv2 as cv
import tensorflow as tf

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def _load_image(image_path, bounding_boxes):    
    img0s = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    im0s = normalize(img0s)
    faces = []
    for det in bounding_boxes:
        h1, h2, w1, w2 = int(det[1]), int(det[3]), int(det[0]), int(det[2])
        h1 = 0 if h1 < 0 else h1
        h2 = 0 if h2 < 0 else h2
        w1 = 0 if w1 < 0 else w1
        w2 = 0 if w2 < 0 else w2
        faces.append(im0s[h1:h2, w1:w2])
    
    resized_faces = [cv.resize(each_face, (160,160)) for each_face in faces]
    return resized_faces


def DataGenerator(image_data, batch_size=64):
    img_paths = []
    img_num = 0
    total_faces = 0
    total_images = len(image_data)
    for key, val in image_data.items():
        img_paths.append(key)
        total_faces += len(val)    
    cache = []

    while True:
        X = []
        X.extend(cache)
        cache = []
        count = len(X)
        while count < batch_size:
            next_img_paths = img_paths[img_num]
            nxt_img_Bbox = image_data[next_img_paths]
            face_images = _load_image(next_img_paths, nxt_img_Bbox)
            num_faces = len(face_images)
            X.extend(face_images)
            count += num_faces
            img_num += 1
            if img_num >= total_images:
                break
        if count > batch_size:
            cache = X[batch_size :]
            X = X[:batch_size]
        
        X = np.array(X)
        yield X 
        if img_num >= total_images:
                break
        

