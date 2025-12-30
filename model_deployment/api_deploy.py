import modal
import os
from fastapi import Request, Response

APP_NAME = "action-recognition-api"
VOLUME_NAME = "dl_a3_dataset"

image = (
    modal.Image.debian_slim()
    .pip_install("tensorflow", "numpy", "opencv-python-headless", "pillow", "fastapi", "uvicorn")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.cls(
    image=image, 
    volumes={"/data": volume}, 
    gpu="T4",           
    scaledown_window=60 
)
class ActionPredictor:
    @modal.enter()
    def load_model(self):
        import tensorflow as tf
        import os
        model_path = "/data/final_model_cnn_lstm.h5"
        classes_path = "/data/classes.txt"
        
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.classes = f.read().splitlines()
        print("Model Loaded!")

    @modal.method()
    def predict_video(self, video_bytes):
        import cv2
        import numpy as np
        import tempfile
        import os
        
        IMG_SIZE = 128
        SEQ_LENGTH = 16
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
            temp_vid.write(video_bytes)
            temp_path = temp_vid.name
            
        frames = []
        cap = cv2.VideoCapture(temp_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                skip = max(int(total_frames / SEQ_LENGTH), 1)
                for i in range(SEQ_LENGTH):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
        finally:
            cap.release()
            if os.path.exists(temp_path): os.remove(temp_path)
            
        while len(frames) < SEQ_LENGTH:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
            
        frames = np.array(frames[:SEQ_LENGTH])
        input_data = np.expand_dims(frames, axis=0)
        
        predictions = self.model.predict(input_data)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return self.classes[class_idx], confidence

@app.function(image=image)
@modal.web_endpoint(method="POST")
async def predict_action_api(request: Request):
    video_data = await request.body()
    
    label, conf = ActionPredictor().predict_video.remote(video_data)
    
    return {
        "action": label, 
        "confidence": float(conf),
        "message": "Success"
    }