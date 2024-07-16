import logging
from PIL import Image
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection


class ObjectDetector:
    def __init__(self, model_name=None, score_threshold=0.1):
        if model_name is None:
            model_name = "google/owlv2-base-patch16"

        # load model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)

        self.text_queries = []
        self.score_threshold = score_threshold

    def add_new_text_query(self, query):
        if len(query) > 0:
            self.text_queries.append(query)

    def detect(self, image):
        '''
        returned bounding box is [x_left, y_bottom, x_right, y_top]
        '''
        
        if len(self.text_queries) == 0:
            logging.warn("Detection requested but text query is empty. Returning empty list.")
            return []
        
        image_file = Image.fromarray(image).convert("RGB")
        detections = []
        with torch.no_grad():
            target_sizes = torch.Tensor([image_file.size[::-1]])
            inputs = self.processor(text=self.text_queries, images=image_file, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            predictions = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                    threshold=self.score_threshold)

            boxes, scores, labels = predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                if score >= self.score_threshold:
                    detections.append((box, label))

        return detections

    def estimate_object_position(self, robot_state, depth, bbox, threshold=2.0):
        if not torch.is_tensor(depth):
            depth = torch.tensor(depth)

        x_robot = robot_state[0]
        y_robot = robot_state[1]
        theta_robot = robot_state[2]

        detection_depth = depth[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
        foreground = detection_depth[detection_depth < threshold].int()

        x = torch.flatten(detection_depth * torch.math.cos(theta_robot) + x_robot)
        y = torch.flatten(detection_depth * torch.math.sin(theta_robot) + y_robot)
        
        x = x[foreground]
        y = y[foreground]

        return x, y