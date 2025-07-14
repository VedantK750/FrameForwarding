import os
import yaml
import numpy as np
from ultralytics import YOLO
import open3d as o3d
import shutil
import cv2


class HumanDepthEstimator:
    """
        Run detection + segmentation + projection + clustering on one frame.

        Returns:
            results[img_name] = {
              'img_file_path: str,               # return full image path
              'depth': float,                    # estimated human depth 
              'human_mask' : np.ndarray,         # binary human mask (H x W) --> full image
              'person_bbox' : np.ndarray,        # [x1,y1,x2,y2] in pixel coords          
            }
        """
    def __init__(self, config_path, args):
        self.args = args
        self.save_dir = args.save_dir
        self.img_dir = args.img_dir
        self.pcd_dir = args.pcd_dir
        self._load_config(config_path)
        self._dir_output_dirs()
        self._load_models()

    def _load_config(self, path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.fx = cfg['camera']['fx']
        self.fy = cfg['camera']['fy']
        self.cx = cfg['camera']['cx']
        self.cy = cfg['camera']['cy']
        self.K = np.array([[self.fx,    0,      self.cx],
                           [0,       self.fy,   self.cy],
                           [0,          0,          1  ]])
        self.T = np.linalg.inv(np.array(cfg['extrinsic_matrix']))

    def _dir_output_dirs(self):
        if os.path.exists(self.save_dir):
            print(f"Removing existing directory: {self.save_dir}")
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_models(self):
        self.det_model = YOLO(self.args.det_model)
        self.seg_model = YOLO(self.args.seg_model)

    def project_pcd_to_image(self, pcd, img):
        # Convert pcd to camera frame, project to 2D
        if pcd.is_empty():
            print(f"Point cloud {self.pcd_dir} is empty. Skipping to the next frame.")
            return None, None, None

        points = np.asarray(pcd.points)
        ones = np.ones((points.shape[0], 1))
        points_hom = np.hstack([points, ones])
        points_cam = (self.T @ points_hom.T).T[:, :3]

        # projecting to image
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        valid_depth = z > 0
        x_valid, y_valid, z_valid = x[valid_depth], y[valid_depth], z[valid_depth]

        u = (self.fx * x_valid / z_valid + self.cx).astype(np.int32)
        v = (self.fy * y_valid / z_valid + self.cy).astype(np.int32)

        h, w = img.shape[:2]
        valid_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_valid, v_valid = u[valid_bounds], v[valid_bounds]
        points_valid = points[valid_depth][valid_bounds]

        if len(points_valid) == 0:
            return None , None, None
        
        return u_valid, v_valid, points_valid

    def filter_human_points(self, u, v, points, mask):
        # Use YOLO mask to filter human points
        if u is None or v is None or points is None:
            return None
        mask_valid = mask[v, u] > 0
        person_points = points[mask_valid]

        if len(person_points) == 0:
            return None
        return person_points

    def DBSCAN_clustering(self, eps, min_points, person_points, frame_name):
        # Apply clustering to 3D human points
        person_pcd = o3d.geometry.PointCloud()
        person_pcd.points = o3d.utility.Vector3dVector(person_points)

        labels = np.array(
            person_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
        )

        # the number of clusters is labels.max() + 1
        max_label = labels.max()
        # print(f"{frame_name}: Found {max_label+1} clusters")

        # if max_label < 0:
        #     print(f"No valid clusters in {frame_name}.")
        #     return None

        clusters = [person_points[np.where(labels == label)[0]] for label in range(max_label + 1)]

        return labels, clusters

    def run(self):
        results = {}
        
        # Main processing loop over frames
        for file_img, file_pcd in zip(sorted(os.listdir(self.img_dir)), sorted(os.listdir(self.pcd_dir))):
            if not file_img.endswith('.png') or not file_pcd.endswith('.pcd'):
                continue

        
            # load image
            full_img_path = os.path.join(self.img_dir, file_img)
            img = cv2.imread(full_img_path)
            if img is None:
                print(f"Could not read image {full_img_path}. Skipping.")
                continue

            # load pcd

            full_pcd_path = os.path.join(self.pcd_dir, file_pcd)
            pcd = o3d.io.read_point_cloud(full_pcd_path)
            scene_points = np.asarray(pcd.points)  # shape (N, 3)

            # run YOLO
            det_result = self.det_model(img)[0]
            person_bbox = None
            # handeling multiple persons by selecting the box with max area
            max_area = -1

            for box in det_result.boxes:
                if int(box.cls[0]) == 0:  # Assuming class 0 is 'person'
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        person_bbox = xyxy.astype(int)

            if person_bbox is None:
                print(f"No person detected in {file_img}. Skipping.")
                continue

            # Extract bounding box coordinates
            x1, y1, x2, y2 = person_bbox
            padding = self.args.padding
            x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(img.shape[1] - 1, x2 + padding), min(img.shape[0] - 1, y2 + padding)

            # draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cropped_img = img[y1:y2, x1:x2]  
            seg_result = self.seg_model.predict(cropped_img, classes=[0], save=False)[0]
            cropped_mask_data = seg_result.masks.data.cpu().numpy()[0] if seg_result.masks else None
            if cropped_mask_data is None:
                print(f"No segmentation mask for {file_img}. Skipping.")
                continue

            crop_h, crop_w = cropped_img.shape[:2]
            resized_crop_mask = cv2.resize(cropped_mask_data, (crop_w, crop_h))

            binary_crop_mask = (resized_crop_mask > self.args.seg_conf).astype(np.uint8)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = binary_crop_mask

            # colour the mask
            colored_mask = np.zeros_like(img)
            colored_mask[mask > 0] = [0, 255, 0] # green
            overlayed_img = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

            overlayed_img_path = os.path.join(self.save_dir, file_img)
            cv2.imwrite(overlayed_img_path, overlayed_img)
            print(f"saved overlayed image: {overlayed_img_path}")

            u_valid, v_valid, points_valid = self.project_pcd_to_image(pcd, img)

            if points_valid  is None:
                print(f"[{file_img}] No valid person points found. SKIPPING FRAME")
                continue

            person_points = self.filter_human_points(u_valid, v_valid, points_valid, mask)

            if person_points is None:
                print(f"No 3D points found for mask in {file_img}. WARNING SKIPPING THIS FRAME")
                continue 


            # Applying DBSCAN clustering

            labels, clusters = self.DBSCAN_clustering(self.args.eps, self.args.min_points, person_points, file_img)

            max_label = labels.max()
            print(f"{file_img}: Found {max_label+1} clusters")

            if max_label < 0:
                print(f"No valid clusters in {file_img}. SKIPPING THIS FRAME")
                continue


            # TODO: Better heuristic for human cluster
            # For now, we assume the largest cluster is the human
            human_cluster = max(clusters, key=lambda c: c.shape[0])
            
            
            # estimating the depth
            
            depths = human_cluster[:, 1]  # y - axis looks like depth here

            depths_sorted = np.sort(depths)
            trim = int(len(depths_sorted) * self.args.trim)

            # trim the 10% of the smallest and largest values and take the median
            if len(depths_sorted) <= 2 * trim:
                print(f"Not enough points to trim in {file_img}. Using all points.")
                median_depth = np.median(depths_sorted)
            else:

                median_depth = np.median(depths_sorted[trim:-trim])
                
            if self.args.vis:  
                results[file_img] = (median_depth, human_cluster, scene_points)
            else:
                results[file_img] = {
                'img_file_path': full_img_path,
                'depth': np.round(median_depth,2),
                'human_mask': mask,
                'person_bbox': person_bbox
                 }
                
        return results

class HumanDepthEstimatorPF(HumanDepthEstimator):
    def __init__(self, config_path, args=None):
        super().__init__(config_path, args)
    
    def _dir_output_dirs(self):

        if not os.path.exists(self.save_dir):
            print(f"[PF] Creating directory:{self.save_dir}")
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            print(f"[PF] Directory already exists: {self.save_dir}")

    def run_on_frame(self, img_path, pcd_path):
        """
        Run depth estimation for a single RGB frame.
        Returns:
            dict: {
                'depth': float,        estimated depth of human from the LiDAR
                'mask': np.ndarray,    returns binary mask of the largest person
                'bbox': np.ndarray,    returns the bbox of the roi
            }
        """
        """
        NOTE: img_path and pcd_path are full path dirs
        """
        result = {}
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            print(f"Empty point cloud: {pcd_path}")
            return None
        scene_points = np.asarray(pcd.points)
        
        img = cv2.imread(img_path)
        det_result = self.det_model(img)[0]
        person_bbox = None
        max_area = -1
        
        for box in det_result.boxes:
                if int(box.cls[0]) == 0:  # Assuming class 0 is 'person'
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        person_bbox = xyxy.astype(int)
                        
        if person_bbox is None:
                print(f"No person detected in {img}. Skipping.")
                return None
        
        x1, y1, x2, y2 = person_bbox
        padding = self.args.padding
        x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(img.shape[1] - 1, x2 + padding), min(img.shape[0] - 1, y2 + padding)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropped_img = img[y1:y2, x1:x2]  
        seg_result = self.seg_model.predict(cropped_img, classes=[0], save=False)[0]
        
        # handeling if there are mutiple person in roi ( taking the biggest mask area )
        if seg_result.masks and len(seg_result.masks.data) > 0: 
            masks_np = seg_result.masks.data.cpu().numpy()
            areas = [np.sum(mask) for mask in masks_np]
            max_idx = np.argmax(areas)
            cropped_mask_data = masks_np[max_idx]
        else:
            return None
        
        if cropped_mask_data is None:
            print(f"No segmentation mask for {img_path}. Skipping.")
            return None

        crop_h, crop_w = cropped_img.shape[:2]
        resized_crop_mask = cv2.resize(cropped_mask_data, (crop_w, crop_h))

        binary_crop_mask = (resized_crop_mask > self.args.seg_conf).astype(np.uint8)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        mask[y1:y2, x1:x2] = binary_crop_mask

        # colour the mask
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = [0, 255, 0] # green
        overlayed_img = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

        # extracting the file name
        img_filename = os.path.basename(img_path)
        overlayed_img_path = os.path.join(self.save_dir, img_filename)
        cv2.imwrite(overlayed_img_path, overlayed_img)
        print(f"saved overlayed image: {overlayed_img_path}")
        
        u_valid, v_valid, points_valid = self.project_pcd_to_image(pcd, img)
        if points_valid  is None:
                print(f"[{img_path}] No valid person points found. SKIPPING FRAME")
                return None
        
        person_points = self.filter_human_points(u_valid, v_valid, points_valid, mask)
        if person_points is None:
                print(f"No 3D Human points found for mask in {img_path}. WARNING SKIPPING THIS FRAME")
                return None 
        
        labels, clusters = self.DBSCAN_clustering(self.args.eps, self.args.min_points, person_points, img_path)
        max_label = labels.max()
        print(f"{img_path}: Found {max_label+1} clusters")

        if max_label < 0:
            print(f"No valid clusters in {img_path}. SKIPPING THIS FRAME")
            return None
        
        # TODO: Better heuristic for human cluster
        # For now, we assume the largest cluster is the human
        human_cluster = max(clusters, key=lambda c: c.shape[0])
        
        
        # estimating the depth
        
        depths = human_cluster[:, 1]  # y - axis looks like depth here

        depths_sorted = np.sort(depths)
        trim = int(len(depths_sorted) * self.args.trim)

        # trim the 10% of the smallest and largest values and take the median
        if len(depths_sorted) <= 2 * trim:
            print(f"Not enough points to trim in {img_path}. Using all points.")
            median_depth = np.median(depths_sorted)
        else:

            median_depth = np.median(depths_sorted[trim:-trim])
            
        result = {'depth': np.round(median_depth,2),
                'human_mask': mask,
                'person_bbox': person_bbox
                 }
        
        return result 
    
    

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process images and point clouds for depth estimation.")
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--pcd_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--det_model', type=str, default='yolo11n.pt')
    parser.add_argument('--seg_model', type=str, default='yolo11m-seg.pt')
    parser.add_argument('--padding', type=int, default=100)
    parser.add_argument('--seg_conf', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--min_points', type=int, default=50)
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config for camera intrinsics/extrinsics")
    parser.add_argument('--trim', type=float, default=0.1)
    parser.add_argument('--vis', action='store_true', help = 'Enable Visualization of point clouds, use only if you are running the script')
    return parser.parse_args()


def main(pf_flag=False):
    
    args = argparse.Namespace(
        img_dir='images',
        pcd_dir='pcd',
        save_dir='overlayed_images',
        det_model='yolo11n.pt',
        seg_model='yolo11m-seg.pt',
        padding=100,
        seg_conf=0.5,
        eps=0.2,
        min_points=50,
        config='config.yaml',
        trim=0.1,
        vis=True  
    )
    
    if pf_flag:
        img_path = '/home/krish/frame_forwarding/images/frame_00016.png'
        pcd_path = '/home/krish/frame_forwarding/pcd/frame_00016.pcd'
        estimator_pf = HumanDepthEstimatorPF(config_path=args.config, args=args)
        result_pf = estimator_pf.run_on_frame(img_path, pcd_path)
        for k, v in result_pf.items(): 
            print(f"{k}: {v}")
    
    else:
        estimator = HumanDepthEstimator(config_path=args.config, args=args)
        results = estimator.run()
        depths = []
        
        for frame_name, (depth, human_cluster, scene_points) in results.items():
            
            print(f"{frame_name}: {depth:.2f}m")
            
            depths.append(depth)
            
            if args.vis:
                human_pcd = o3d.geometry.PointCloud()
                human_pcd.points = o3d.utility.Vector3dVector(human_cluster)
                human_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red

                scene_pcd = o3d.geometry.PointCloud()
                scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
                scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
                
                o3d.visualization.draw_geometries(
                [scene_pcd, human_pcd], window_name=f"Scene with Human Cluster - {frame_name}")
                
        print(f"Depths : {depths}")
            
if __name__ == "__main__":
    # run this script if you want to visualize the point clouds
    main(pf_flag=False)