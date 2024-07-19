import logging
from PIL import Image
import torch
import numpy as np
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

        self.text_queries = ["don't drive over the rug"]
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

    def world_to_pixel(self, x, y, z, robot_state):
        K = self.get_camera_intrinsics_mat()
        T = self.get_camera_extrinsics_mat(robot_state=robot_state)
        pw = np.array([x, y, z, 1])
        pi = np.matmul(T, pw)
        pp = np.matmul(K, pi[:3])
        return pp[0]/pp[2], pp[1]/pp[2]
    
    def pixel_to_world(self, uv, K_inv, T_inv):
        # normalize pixel coordinates
        xy_img = np.matmul(K_inv, uv)
        # transform to world frame
        xy_img = np.vstack((xy_img, np.ones(xy_img.shape[1])))
        xyz = np.matmul(T_inv, xy_img)
        x = xyz[0]
        y = xyz[1]
        return x, y
    
    def estimate_object_position(self, depth, bbox, K_inv, T_inv, threshold=3.0, height=256, width=256):
        # get indexes of bounding box pixels
        try:
            bbox = bbox.cpu().numpy()
        except AttributeError as e: # just in case it is not a torch tensor
            bbox = np.array(bbox)

        # height, width = self.sensors_specs['camera'].resolution
        u_lim = np.clip([bbox[0], bbox[2]], 0, width).astype(int)
        v_lim = np.clip([bbox[1], bbox[3]], 0, height).astype(int)
        u, v = np.meshgrid(np.arange(u_lim[0], u_lim[1]), np.arange(v_lim[0], v_lim[1]))
        u = u.flatten()
        v = v.flatten()
        # get depth for each pixel in the bounding box
        depth_obj = depth[v_lim[0]:v_lim[1], u_lim[0]:u_lim[1]]
        depth_x = depth_obj.flatten()
        depth_y = depth_obj.T.flatten()
        # add depth information
        u = depth_x * u
        v = depth_y * v
        ones = depth_x * np.ones(u.shape).flatten()

        # only consider pixels below depth threshold
        iu = np.where(depth_x < threshold)[0]
        iv = np.where(depth_y < threshold)[0]
        uv = np.vstack((u[iu], v[iv], ones[iu]))
        # pixel -> world
        x, y = self.pixel_to_world(uv, K_inv, T_inv)
        return x, y