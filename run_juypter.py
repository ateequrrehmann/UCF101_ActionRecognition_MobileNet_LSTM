import modal
import os
import subprocess

APP_NAME = "action-recog-jupyter"
VOLUME_NAME = "dl_a3_dataset"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "tensorflow[and-cuda]",     
        "numpy", 
        "opencv-python-headless", 
        "matplotlib", 
        "scikit-learn", 
        "jupyterlab", 
        "tqdm"
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .env({"CUDNN_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn"})
)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400,
    gpu="A10G"      
)
def start_jupyter():
    import tensorflow as tf
    import os
    
    print(f"\n\nSTARTING JUPYTER LAB ON CLOUD GPU 🚀")
    
    print("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"SUCCESS! Detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"   - {gpu}")
        print(f"TensorFlow Version: {tf.__version__}")
    else:
        print("WARNING: No GPU detected via TensorFlow.")
            
        try:
            print("\nRunning nvidia-smi check:")
            os.system("nvidia-smi")
        except:
            pass

    print("-" * 40)

    with modal.forward(8888) as tunnel:
        print(f"🔗 CLICK THIS LINK: {tunnel.url}")
        print("-" * 40 + "\n")
        
            
        subprocess.run(
            [
                "jupyter", "lab",
                "--no-browser",
                "--ip=0.0.0.0",
                "--port=8888",
                "--allow-root",
                "--notebook-dir=/root", 
                "--NotebookApp.token="
            ],
            check=True
        )

@app.local_entrypoint()
def main():
    start_jupyter.remote()