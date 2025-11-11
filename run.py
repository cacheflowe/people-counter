import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import argparse
import torch
from datetime import datetime
import os
import time

# Check for CUDA device and set it

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
print(f'torch.__version__: {torch.__version__}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.get_device_name(0)): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')

# Track unique people seen
seen_tracker_ids = set()


def get_daily_count_file():
	"""Generate the filename for today's count file"""
	today = datetime.now().strftime("%Y-%m-%d")
	output_dir = "output"
	os.makedirs(output_dir, exist_ok=True)
	return os.path.join(output_dir, f"people_count_{today}.txt")


def load_daily_count():
	"""Load today's people count if it exists"""
	filepath = get_daily_count_file()
	
	if os.path.exists(filepath):
		try:
			with open(filepath, 'r') as f:
				count = int(f.read().strip())
				print(f"Loaded count of {count} from {filepath}")
				return count
		except Exception as e:
			print(f"Error loading daily count: {e}")
			return 0
	return 0


def save_daily_count():
	"""Save the current people count to today's file"""
	filepath = get_daily_count_file()
	
	try:
		with open(filepath, 'w') as f:
			f.write(str(len(seen_tracker_ids)))
		print(f"Saved count to {filepath}")
	except Exception as e:
		print(f"Error saving daily count: {e}")


def parse_args():
	parser = argparse.ArgumentParser(description="Real-time person tracking and counting")
	parser.add_argument("--webcam-resolution", nargs=2, type=int, default=[1280, 720], help="Webcam resolution")
	parser.add_argument("--camera", type=int, default=0, help="Camera index")
	parser.add_argument("--model", type=str, default="datasets/yolov8n.pt", help="Path to YOLO model")
	return parser.parse_args()


def callback(frame: np.ndarray, model: YOLO, tracker: sv.ByteTrack,
			box_annotator: sv.BoxAnnotator, label_annotator: sv.LabelAnnotator,
			trace_annotator: sv.TraceAnnotator) -> np.ndarray:
	global seen_tracker_ids

	# inference with verbose=False to suppress output and half=True for FP16 on GPU
	results = model(frame, verbose=False, half=(device == 'cuda'))[0]
	detections = sv.Detections.from_ultralytics(results)

	# Filter for person class only (class_id 0 in COCO dataset)
	detections = detections[detections.class_id == 0]
	detections = tracker.update_with_detections(detections)

	# Track unique people
	if detections.tracker_id is not None:
		for tracker_id in detections.tracker_id:
			seen_tracker_ids.add(tracker_id)

	# Prepare labels - id for unique people
	labels = [
		f"#{tracker_id} {model.model.names[class_id]}"
		for class_id, tracker_id
		in zip(detections.class_id, detections.tracker_id)
	]

	annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
	annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
	annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

	# Add total people count to frame
	cv2.putText(
		annotated_frame,
		f"Total People Seen: {len(seen_tracker_ids)}",
		(10, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.5,
		(0, 0, 0),
		2
	)

	return annotated_frame


def main():
	args = parse_args()
	frame_width, frame_height = args.webcam_resolution

	# Load existing count for today
	initial_count = load_daily_count()
	if initial_count > 0:
		print(f"Resuming today's count: {initial_count} people already seen")

	# Initialize model on specified device with optimizations
	model = YOLO(args.model).to(device)
	# Default is 30, reduce to 15-20 if needed
	tracker = sv.ByteTrack(frame_rate=30)
	
	# Set smaller inference size for faster processing (default is 640)
	# You can adjust this - smaller = faster but less accurate
	model.overrides['imgsz'] = 416  # Try 320, 416, or 640
	
	# Use BoundingBoxAnnotator instead of BoxAnnotator for consistency
	box_annotator = sv.BoxAnnotator(thickness=2)
	label_annotator = sv.LabelAnnotator()
	trace_annotator = sv.TraceAnnotator()

	# Setup webcam with specified resolution
	cap = cv2.VideoCapture(args.camera)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

	print(f"Using camera {args.camera} at {frame_width}x{frame_height}")
	print("Press 'q' or ESC to quit")

	# Track last save time for periodic saving
	last_save_time = time.time()
	save_interval = 60  # Save every 60 seconds

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Failed to grab frame")
			break

		annotated_frame = callback(frame, model, tracker, box_annotator, label_annotator, trace_annotator)

		cv2.imshow("Supervision People Counter", annotated_frame)

		# Save count every minute
		current_time = time.time()
		if current_time - last_save_time >= save_interval:
			save_daily_count()
			last_save_time = current_time

		# Allow both 'q' and ESC to quit
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q') or key == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

	# Save the final count
	save_daily_count()
	print(f"\nTotal unique people seen today: {len(seen_tracker_ids)}")


if __name__ == "__main__":
	main()
