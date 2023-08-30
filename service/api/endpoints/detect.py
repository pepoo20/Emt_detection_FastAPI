from fastapi import APIRouter, UploadFile,HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.logic.onnx_inference import emotion_detector
from service.core.schemas.output import APIOutPut
detect_router = APIRouter()

@detect_router.post("/detect/", response_model=APIOutPut)
def detect(im: UploadFile):
    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)

    if im.filename.split(".")[-1] in ("jpg" , "png" , "jpeg"):
        pass
    else:
        raise HTTPException(
            status_code=415 , detail= "not an image"
        )

    return emotion_detector(image)
