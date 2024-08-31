#python inference_on_a_video.py -i /data/test.mp4 -o /data/out/ -t "person" -bt 0.4 -tt 0.25
from groundingdino.util.inference import load_model, predict, annotate, Model
import cv2
import os
import argparse
import torch
from torchvision.ops import box_convert
from PIL import Image
import numpy as np
import groundingdino.datasets.transforms as T

class Detection:
    def __init__(self, object_name : str, bbox_xywh : np.array) -> None:
        self.object_name = object_name
        self.bbox_xywh = bbox_xywh
    
    def __repr__(self) -> str:
        bbox_values = self.bbox_xywh.flatten()
        return f"{self.object_name} - {','.join(map(str, bbox_values))}\n"
        
        
class BBoxConfig:
    def __init__(self, path : str) -> None:
        #if not os.path.isfile(path):
        #    raise ValueError("Invalid path was provided for the config file")
        
        self.file = open(path, "w+")
        self.frame_idx = 0
    
    def set_frame(self, new_frame_idx : int):
        self.frame_idx = new_frame_idx
    
    def add_frame(self, detections):
        self.file.write(f"{self.frame_idx}:\n")
        self.frame_idx += 1
        for detection in detections:
            self.file.write(str(detection))
    
    def __del__(self):
        self.file.close()

def annotate_video(input_video_path, output_dir, model,
    text_prompt, box_threshold, text_threshold):
    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_extension = ".avi"

    out_video_path = os.path.join(output_dir, f"annotated_video{video_extension}")
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    out_config_path = os.path.join(output_dir, f"annotations.txt")
    out_config = BBoxConfig(out_config_path)

    
    current_frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        preprocessed_image = Model.preprocess_image(frame)
            
        boxes, logits, phrases = predict(
            model=model,
            image=preprocessed_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold)
            
        boxes_cxcywh = boxes * torch.Tensor([width, height, width, height])
        boxes_xywh = box_convert(boxes=boxes_cxcywh, in_fmt="cxcywh", out_fmt="xywh").numpy()
        detections = []
        for phrase, box in zip(phrases, boxes_xywh):
            detections.append(Detection(phrase, box))
        out_config.add_frame(detections)
            
        annotated_image = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        
        out_video.write(annotated_frame)
            
        if current_frame_index % (total_frame_count // 10) == 0:
            print(f"Processed {current_frame_index / total_frame_count:.0%} of the frames")
        
        current_frame_index += 1

    print(f"Finished processing, files written to {out_video_path} and {out_config_path}")
    cap.release()
    out_video.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--input", "-i", type=str, required=True, help="path to video file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", "-bt", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", "-tt", type=float, default=0.25, help="text threshold")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only! default=False")
    args = parser.parse_args()
    
    # cfg
    input_path = args.input
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model("/opt/program/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "/opt/program/weights/groundingdino_swint_ogc.pth", device="cpu" if args.cpu_only else "cuda")
        
    annotate_video(input_path, output_dir, model, text_prompt, box_threshold, text_threshold)
