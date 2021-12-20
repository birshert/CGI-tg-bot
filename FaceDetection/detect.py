import cv2
from PIL import Image


def detect_img(model, img):
    faces = model.detect(img)
    if type(faces[0]) == "NoneType":
        return [False, None]
    else:
        return [True, faces[0]]


def detect_video(model, video):
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    faces = []
    for frame in frames:
        boxes, _ = model.detect(frame)
        if type(boxes) == "NoneType":
            faces.append([False, None])
        else:
            faces.append([True, boxes])
    return faces
