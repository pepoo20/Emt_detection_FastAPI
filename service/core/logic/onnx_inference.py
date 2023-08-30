import onnxruntime as rt
import cv2
import numpy as np


def emotion_detector(img_array):
    providers = ['CPUExecutionProvider']
    if len(img_array.shape)==2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    m_q = rt.InferenceSession("service\core\logic\pretrained_model_quantized.onnx",providers = providers)
    
    test_image = cv2.resize(img_array, (256,256))
    im = np.float32(test_image)
    img_array  = np.expand_dims(im , axis = 0)
    print(img_array.shape)

    onnx_pred = m_q.run(['dense_5'], {"input": img_array})
    print(np.argmax(onnx_pred[0][0]))

    emotion = ""
    if np.argmax(onnx_pred[0][0])==0:
        emotion = "angry"
    elif np.argmax(onnx_pred[0][0])==1:
        emotion = "happy"
    else:
        emotion = "sad"
    return {"emotion": emotion}