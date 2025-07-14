# Frame Forwarding

Frame Forwarding Repository for **DARPA Triage Challenge Phase 2 – 2025**  

This project processes synchronized RGB and point cloud (PCD) frames to estimate human depth, extract the largest segmentation mask, and forward the best representative frame in temporal windows.



##  Environment Setup

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/VedantK750/FrameForwarding.git
cd FrameForwarding
```

### 2. Create  environment

```bash
conda env create -f environment.yml
conda activate frameforward
```

## Running the Demo

You can run the main demo using:

```bash
python demo.py \
    --img_dir path/to/images \
    --pcd_dir path/to/pointclouds \
    --save_dir overlayed_images \
    --config path/to/cam_intrinsics.yaml \
    --save_best_dir output_best_frames \
    --WINDOW_SIZE 250 \
    --MIN_MASK_AREA 5000 \
```



