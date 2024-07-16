import os, sys
import requests
from PIL import Image
import numpy as np
import torch
import time
import json
import cv2
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pathlib import Path

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path)  # add lang-reachability to PYTHONPATH

# load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

# params

test_idx = 2
score_threshold = 0.3
# scene_map = {0:"102344469", 1:"102344022", 2:"102344094", 3:"103997403_171030405", 4:"102815859", 5:"102816216"}
scene_map = {0: "00099-226REUyJh2K", 1: "00013-sfbj7jspYWj", 2: "00198-eZrc5fLTCmi"}
text_queries = {0:["don't crash into the couch", "don't drive over the rug", "stay away from the piano"],
                1:["avoid the plants on the ground", "stay away from the sofa"],
                2:["don't drive over the rug", "avoid the plants on the ground", "stay away from the sofa"]}
save_frames = False
save_video = True
save_json = False
save_partial = False


# load video
data_dir = os.path.join(dir_path, 'tests', 'outputs', f"{scene_map[test_idx]}")
video_name = f"{scene_map[test_idx]}_rgb.mp4"
video_str = video_name.split('.')[0]
vidcap_path = os.path.join(data_dir, video_name)
vidcap = cv2.VideoCapture(vidcap_path)
success, frame = vidcap.read()
if not success:
    print('error reading video')
    exit()

image_file = Image.fromarray(frame)
height, width, layers = frame.shape
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

# configure saving paths
save_dir = os.path.join(dir_path, 'tests', 'results', f'scene_{test_idx}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if save_frames:
    save_frame_dir = os.path.join(save_dir, 'frames')
    if not os.path.exists(save_frame_dir):
        os.makedirs(save_frame_dir)
if save_json:
    save_json_dir = os.path.join(save_dir, 'json')
    if not os.path.exists(save_json_dir):
        os.makedirs(save_json_dir)

# save video
vidres_name = os.path.join(save_dir, f"scene_{test_idx}_st_{int(100*score_threshold)}_det.avi")
vidres = cv2.VideoWriter(vidres_name, 0, 10, (width, height))

# predict frame by frame and save result
text_query = text_queries[test_idx]
result = {"annotations": {}}
count = 0
step = 1
print(f'scene: {test_idx}')
print(f'query: {text_query}')
try:
    while success:
        if count % step == 0:
            frame_result = []
            image_file_np = np.array(image_file)
            image_file_np = cv2.cvtColor(image_file_np, cv2.COLOR_BGR2RGB)  # Change from BGR to RGB

            with torch.no_grad():
                target_sizes = torch.Tensor([image_file.size[::-1]])
                start_time = time.time()
                inputs = processor(text=text_query, images=image_file, return_tensors="pt").to(device)
                model = model.to(device)
                outputs = model(**inputs)
                predictions = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                      threshold=score_threshold)
                end_time = time.time()

                # print(f'inference time: {end_time - start_time}')
                print(f"\rprogress: {count}/{length} frames ({100*count/length:.2f}%)", end="", flush=True)
                image_np = image_file_np.copy()  # Work with a copy of the image
                boxes, scores, labels = predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]
                for box, score, label in zip(boxes, scores, labels):
                    # print(f"label: {text_query[label]}, score: {score:.2f}")
                    if score >= score_threshold:
                        cv2.rectangle(image_np, (int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3])), (255, 0, 0), 2)
                        cv2.putText(image_np, f"{text_query[label]}",
                                    (int(box[0]), int(box[1]) - 30),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.putText(image_np, f"{score:.2f}",
                                    (int(box[0]), int(box[1])),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        detection = {text_query[label]: box.tolist()}
                        frame_result.append(detection)

                if save_frames:
                    cv2.imwrite(os.path.join(save_frame_dir, f'frame{int(count)}.jpg'), frame)
                    # cv2.imwrite(f'/home/leo/git/lang-reachability/results/lab-video-{int(count / 2)}.jpg', image_np)
                if save_video:
                    vidframe = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    vidres.write(vidframe)
                if save_json:
                    result["annotations"][f"{count}"] = frame_result

        success, frame = vidcap.read()
        image_file = Image.fromarray(frame)
        count += 1

    if save_json:
        with open(os.path.join(save_json_dir, f'results.json'), "w") as outfile:
            json.dump(result, outfile, indent=4)

    if save_video:
        print(f'video with detections saved to {vidres_name}')

except KeyboardInterrupt:
    # Convert and write JSON object to file
    if save_partial and save_json:
        print('saving partial result')
        with open(os.path.join(save_json_dir, f'results.json'), "w") as outfile:
            json.dump(result, outfile, indent=4)

except Exception:
    if save_json:
        with open(os.path.join(save_json_dir, f'results.json'), "w") as outfile:
            json.dump(result, outfile, indent=4)