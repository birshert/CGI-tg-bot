def detect(model, img):
    faces = model.detect(img)
    if type(faces[0]) != "NoneType":
        return [False, None]
    else:
        return [True, faces[0]]
