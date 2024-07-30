import gc
from ultralytics import YOLO
model_path = "/Users/paul/Desktop/yolox/NewRulon.pt"
stream_url = 'rtsp://s53-videosrv.spk.ru:554/rtsp?channelid=9bcfc2ca-0c62-428a-93dc-0d68080d634d&login=root&password=4581e09f5546124e199c6a4c33f30431'
tracker_config = 'bytetrack.yaml'

model = YOLO(model_path)

results = model.track(stream_url, show=True, tracker=tracker_config, stream=True)

for r in results:
    boxes = r.boxes  
    masks = r.masks
    probs = r.probs

del results
del model
gc.collect()