import cv2
import numpy as np
import os
from ultralytics import YOLO
from depth_estimator import HumanDepthEstimatorPF
import argparse 
import shutil

# input
PCD = '/home/krish/frame_forwarding/pcd'
IMG = '/home/krish/frame_forwarding/images'
CONFIG = '/home/krish/frame_forwarding/config.yaml'

# VIDEO_PATH = "/home/krish/Downloads/video_20250708_201255_edit(1).mp4"  # <--- CHANGE THIS to your input video file

# output
OUTPUT_MASK_FOLDER = "output_masks"
OUTPUT_BEST_FRAME_FOLDER = "output_best_frames"


# constants
TEMPORAL_WINDOW_SIZE = 25          
MIN_MASK_AREA_THRESHOLD = 5000      
CONFIDENCE_THRESHOLD = 0.5          


def setup_directories():
    """Creates the necessary output directories if they don't exist."""
    if os.path.exists(OUTPUT_MASK_FOLDER) and os.path.exists(OUTPUT_BEST_FRAME_FOLDER):
        print(f"[demo] Removing existing dirs {OUTPUT_MASK_FOLDER} and {OUTPUT_BEST_FRAME_FOLDER}")
        shutil.rmtree(OUTPUT_MASK_FOLDER)
        shutil.rmtree(OUTPUT_BEST_FRAME_FOLDER)
        
    os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_BEST_FRAME_FOLDER, exist_ok=True)
    print(f"Output directories '{OUTPUT_MASK_FOLDER}' and '{OUTPUT_BEST_FRAME_FOLDER}' are ready.")

# def load_model():
#     """Loads the YOLOv8 segmentation model."""
#     print(f"Loading YOLO model: {YOLO_MODEL_PATH}...")
#     model = YOLO(YOLO_MODEL_PATH)
#     print("Model loaded successfully.")
#     return model



def find_largest_human_mask(mask):
    """
    Processes YOLO results for a single frame to find the largest human mask.
    
    Args:
        results: The results from the YOLO model for a single frame.
        frame_shape: The (height, width) of the original frame.

    Returns:
        A tuple (mask, area) or (None, 0) if no valid human is found.
        - mask: A 2D numpy array (binary mask, 0 or 255).
        - area: The area of the mask in pixels.
    """
    
    # Check if any masks were detected
    if mask is None:
        return None, 0
    
    area = np.sum(mask)
    if area < MIN_MASK_AREA_THRESHOLD:
        print("Skipped as not meeting area criteria")
        return None, 0 
    
    return mask, area


def process_and_clear_window(window_data, window_count):
    """
    Finds the frame with the largest mask in the window, saves it, and clears the window.
    """
    if not window_data:
        return

    best_entry = max(window_data, key=lambda item: item[1])
    best_frame, best_area, best_frame_index = best_entry

    output_filename = os.path.join(OUTPUT_BEST_FRAME_FOLDER, f"best_frame_window_{window_count:03d}_frame_{best_frame_index:04d}.jpg")
    cv2.imwrite(output_filename, best_frame)
    
    print(f"\n--- Temporal Window {window_count} Processed ---")
    print(f"Found best frame at index {best_frame_index} with mask area {int(best_area)}.")
    print(f"Saved to: {output_filename}\n")
    
    window_data.clear()



def parse_args():
    parser = argparse.ArgumentParser(description="Frame Forwarding Pipeline")
    parser.add_argument('--img_dir', type=str, default = IMG)
    parser.add_argument('--pcd_dir', type=str, default = PCD)
    parser.add_argument('--save_dir', type=str, default=OUTPUT_MASK_FOLDER)
    parser.add_argument('--save_best_dir', type=str, default=OUTPUT_BEST_FRAME_FOLDER)
    parser.add_argument('--config', type=str, default =CONFIG)
    parser.add_argument('--det_model', type=str, default='yolo11n.pt')
    parser.add_argument('--seg_model', type=str, default='yolo11m-seg.pt')
    parser.add_argument('--seg_conf', type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument('--padding', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--min_points', type=int, default=50)
    parser.add_argument('--trim', type=float, default=0.1)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--WINDOW_SIZE', type=float, default=TEMPORAL_WINDOW_SIZE)
    parser.add_argument('--MIN_MASK_AREA', type=float, default=MIN_MASK_AREA_THRESHOLD)
    return parser.parse_args()



def main():
    args = parse_args()
    
    estimator_pf = HumanDepthEstimatorPF(config_path=args.config, args=args)
    setup_directories()
    

    frame_index = 0
    temporal_window_data = [] # List to store (frame, area, frame_index) for the current window
    window_count = 1

    for file_img, file_pcd in zip(sorted(os.listdir(args.img_dir)), sorted(os.listdir(args.pcd_dir))):

        # Find the largest human mask that meets the threshold
        full_img_path = os.path.join(args.img_dir, file_img)
        full_pcd_path = os.path.join(args.pcd_dir, file_pcd)
        
        result_pf = estimator_pf.run_on_frame(full_img_path, full_pcd_path)
        if result_pf is None:
            print(f"WARNING!!! Skipping frame {frame_index} due to invalid result.")
            frame_index += 1
            continue
        mask =result_pf["human_mask"]
        depth = result_pf["depth"]

        img = cv2.imread(full_img_path)
        
        largest_mask, area = find_largest_human_mask(mask)
        
        area_depth_normalized = np.round(area/depth,3 )

        # If a valid mask was found
        if largest_mask is not None:
            print(f"Frame {frame_index}: Found human mask with normalized area {int(area_depth_normalized)} with estimated depth as {depth} m.")
            
            # Save the individual mask
            mask_filename = os.path.join(OUTPUT_MASK_FOLDER, f"mask_{frame_index:04d}.png")
            cv2.imwrite(mask_filename, largest_mask)
            
            # Add data to temporal window
            # store a copy of the frame to avoid reference issues
            temporal_window_data.append((img.copy(), area_depth_normalized, frame_index))

        # Check if the temporal window is full
        if len(temporal_window_data) >= TEMPORAL_WINDOW_SIZE:
            process_and_clear_window(temporal_window_data, window_count)
            window_count += 1

        frame_index += 1

    # process any remaining frames in the last (incomplete) window
    if temporal_window_data:
        print("Processing the final batch of frames...")
        process_and_clear_window(temporal_window_data, window_count)

    
    
    print("Processing complete.")

if __name__ == "__main__":
    main()