import cv2
import mediapipe as mp
import os
mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

def faceDetection(path, face_detection):
    imageFaceInfo = {}
    unclassifiedImages = []
    # Fetching all the files in the folder
    files = []
    for (dirpath, _, filenames) in os.walk(path):
        files.extend([os.path.join(dirpath, name) for name in filenames])
    # For the time being
    # files = path
    total = len(files)
    for idx, file in enumerate(files):
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(idx,"/",total)
        h,w,_ = image.shape
        # Draw face detections of each face.
        if not results.detections:
            unclassifiedImages.append(file)
            continue
        tmp = []
        for detection in results.detections:
            data = detection.location_data.relative_bounding_box
            xmin, ymin, width, height = data.xmin * w, data.ymin *h, data.width *w, data.height *h
            tmp.append([xmin, ymin, xmin+width, ymin+height])
        imageFaceInfo[file] = tmp
    return imageFaceInfo, unclassifiedImages
                            
                
