# People Counter

A simple people counter using YOLOv8 and Supervision library. Use a webcam as input to track and count people in real-time.

## Install :

Intall a compatible version of PyTorch with CUDA support before installing other dependencies. For example, for CUDA 11.8:

Then install other dependencies:

```powershell
python -m venv .venv
.venv\\Scripts\\activate.ps1  # (Windows/Powershell)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118 
pip install supervision[desktop] ultralytics
```

## Usage:

```powershell
.venv\\Scripts\\activate.ps1
python run.py --webcam-resolution 1280 720 --camera 1 --confidence 0.6
```

## Notes :

- https://supervision.roboflow.com/latest/how_to/track_objects/
- https://github.com/roboflow/supervision/tree/develop/examples
- https://github.com/ultralytics/ultralytics/issues/3084

Download yolo model:

```
ultralytics download yolov8n
```

