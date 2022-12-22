import time
import torch
from ML.yolov7face.models.experimental import attempt_load
from ML.yolov7face.utils.datasets import LoadImages
from ML.yolov7face.utils.general import check_img_size, set_logging
from ML.yolov7face.utils.torch_utils import select_device
from ML.yolov7face.face_detection import run_inference
from ML.FaceNet.model import get_pretrained_model
from ML.FaceNet.dataGenerator import DataGenerator
from ML.MediaPipe.face_detection import faceDetection
import mediapipe as mp
from sklearn.preprocessing import Normalizer



def detectPeopleYolov7(opt):
    imageFaceInfo = {}
    unclassified_img = []
    # Get the input folder for images in source
    source, weights, imgsz, kpt_label = opt["source"], opt["weights"], opt["img_size"], opt["kpt_label"]
    # Initialize
    set_logging()
    device = select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    dataset = LoadImages(source, img_size=imgsz, stride=stride) 
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        boundingBox = run_inference(model, img, opt["conf_thres"], opt["iou_thres"], opt["classes"], opt["agnostic_nms"], kpt_label, path, im0s, dataset, names)
        # imageFaceInfo[path] = np.array([im0s[int(det[1]):int(det[3]), int(det[0]):int(det[2])]  for det in boundingBox])
        if len(boundingBox) == 0: # No person is detected in the image
            unclassified_img.append(path)
            continue
        imageFaceInfo[path] = boundingBox
    print(f'Done. ({time.time() - t0:.3f}s)')
    return imageFaceInfo, unclassified_img


def detectPeopleMediaPipe(opt):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=opt["confidence"])
    return faceDetection(opt["path"], face_detection)


def generateEmbedding(dic,batch_size=64):
    faceNet = get_pretrained_model()
    data = faceNet.predict(
        DataGenerator(dic, batch_size=64),
        batch_size=batch_size,
        verbose='auto',
        workers=-1,
        use_multiprocessing=True
    )
    transformer = Normalizer().fit(data)
    return transformer.transform(data)
