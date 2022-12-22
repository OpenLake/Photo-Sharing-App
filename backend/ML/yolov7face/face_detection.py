from pathlib import Path
from ML.yolov7face.utils.datasets import LoadImages
from ML.yolov7face.utils.general import non_max_suppression, scale_coords
from ML.yolov7face.utils.torch_utils import time_synchronized

def run_inference(model, img, conf_thres, iou_thres, classes, agnostic_nms, kpt_label, path, im0s, dataset, names):
    # This function run the image via the model and return the bounding box coordinates of the model
    boundingBox = []
    # t1 = time_synchronized()
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label)
    # t2 = time_synchronized()
    # Process detections
    for _, det in enumerate(pred):  # detections per image
        p, s, im0, _ = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        s += '%gx%g ' % img.shape[2:]  # print string
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)
            # Write results
            for (*xyxy, _, cls) in reversed(det[:,:6]):
                # c = int(cls)  # integer class
                boundingBox.append(xyxy)

        return boundingBox
