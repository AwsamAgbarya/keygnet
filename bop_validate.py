"""
BOP Dataset Validation and Visualization Utilities

Verify dataset integrity and visualize generated frames with poses.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings('ignore')

try:
    import cv2
    from PIL import Image
    import trimesh
except ImportError as e:
    print(f"Warning: Missing dependency. Install: pip install opencv-python pillow trimesh")


class BOPDatasetValidator:
    """Validate BOP dataset structure and content."""
    
    def __init__(self, dataset_dir, split='train', obj_id=1):
        self.split = split
        self.obj_id = obj_id
        self.dataset_dir = Path(dataset_dir)
        self.results = {}
        
    def validate_structure(self) -> bool:
        """Check if dataset has correct BOP structure."""
        print("Validating dataset structure...")
        
        required_dirs = [
            "models",
            f"{self.split}/{self.obj_id}",
            f"{self.split}/{self.obj_id}/rgb",
            f"{self.split}/{self.obj_id}/depth",
            f"{self.split}/{self.obj_id}/mask",
            f"{self.split}/{self.obj_id}/mask_visib",
        ]
        
        required_files = [
            "camera.json",
            "dataset_info.json",
            "models/models_info.json",
            f"{self.split}/{self.obj_id}/scene_gt.json",
            f"{self.split}/{self.obj_id}/scene_camera.json",
        ]
        
        all_valid = True
        
        for dir_name in required_dirs:
            path = self.dataset_dir / dir_name
            if path.exists():
                print(f"  ✓ {dir_name}")
            else:
                print(f"  ✗ {dir_name} MISSING")
                all_valid = False
        
        for file_name in required_files:
            path = self.dataset_dir / file_name
            if path.exists():
                print(f"  ✓ {file_name}")
            else:
                print(f"  ✗ {file_name} MISSING")
                all_valid = False
        
        self.results['structure_valid'] = all_valid
        return all_valid
    
    def validate_annotations(self) -> bool:
        """Check annotation files are valid JSON with correct format."""
        print("\nValidating annotations...")
        
        try:
            # Load scene_gt.json
            with open(self.dataset_dir / f"{self.split}/{self.obj_id}/scene_gt.json") as f:
                scene_gt = json.load(f)
            
            num_frames = len(scene_gt)
            print(f"  ✓ scene_gt.json loaded ({num_frames} frames)")
            
            # Validate first annotation
            first_anno = scene_gt[0][0]  # First frame, first object
            required_keys = {'cam_R_m2c', 'cam_t_m2c', 'obj_id', 'cam_K'}
            
            if all(k in first_anno for k in required_keys):
                print(f"  ✓ Annotation format valid")
                
                # Check pose dimensions
                R = np.array(first_anno['cam_R_m2c']).reshape(3, 3)
                t = np.array(first_anno['cam_t_m2c']).reshape(3, 1)
                K = np.array(first_anno['cam_K']).reshape(3, 3)
                
                # Verify rotation is orthogonal
                det = np.linalg.det(R)
                if abs(det - 1.0) < 0.01:
                    print(f"  ✓ Rotation matrix valid (det={det:.4f})")
                else:
                    print(f"  ⚠ Rotation matrix det={det:.4f} (expected ~1.0)")
            else:
                print(f"  ✗ Missing keys in annotation: {required_keys - set(first_anno.keys())}")
                return False
            
            # Load scene_camera.json
            with open(self.dataset_dir / f"{self.split}/{self.obj_id}/scene_camera.json") as f:
                scene_camera = json.load(f)
            
            if len(scene_camera) == num_frames:
                print(f"  ✓ scene_camera.json has {len(scene_camera)} entries")
            else:
                print(f"  ⚠ scene_camera.json entries ({len(scene_camera)}) != frames ({num_frames})")
            
            self.results['annotations_valid'] = True
            return True
            
        except Exception as e:
            print(f"  ✗ Error loading annotations: {e}")
            self.results['annotations_valid'] = False
            return False
    
    def validate_images(self) -> bool:
        """Check image files exist and have correct dimensions."""
        print("\nValidating image files...")
        
        try:
            rgb_dir = self.dataset_dir / f"{self.split}/{self.obj_id}/rgb"
            depth_dir = self.dataset_dir / f"{self.split}/{self.obj_id}/depth"
            
            rgb_files = sorted(rgb_dir.glob("*.png"))
            depth_files = sorted(depth_dir.glob("*.png"))
            
            print(f"  Found {len(rgb_files)} RGB images")
            print(f"  Found {len(depth_files)} depth images")
            
            if len(rgb_files) == 0:
                print(f"  ✗ No RGB images found")
                return False
            
            # Check first RGB image
            rgb = Image.open(rgb_files[0])
            print(f"  ✓ RGB image shape: {rgb.size}")
            
            # Check first depth image
            depth = Image.open(depth_files[0])
            print(f"  ✓ Depth image shape: {depth.size}")
            
            # Verify matching counts
            if len(rgb_files) == len(depth_files):
                print(f"  ✓ RGB and depth counts match")
                self.results['images_valid'] = True
                return True
            else:
                print(f"  ✗ RGB/depth count mismatch")
                return False
                
        except Exception as e:
            print(f"  ✗ Error validating images: {e}")
            return False
    
    def validate_model(self) -> bool:
        """Check 3D model file."""
        print("\nValidating 3D model...")
        
        try:
            model_path = self.dataset_dir / "models/obj_000001.ply"
            if not model_path.exists():
                print(f"  ✗ Model file not found: {model_path}")
                return False
            
            mesh = trimesh.load(str(model_path))
            print(f"  ✓ Model loaded")
            print(f"    - Vertices: {len(mesh.vertices)}")
            print(f"    - Faces: {len(mesh.faces)}")
            print(f"    - Bounds: {mesh.bounds}")
            
            self.results['model_valid'] = True
            return True
            
        except Exception as e:
            print(f"  ✗ Error validating model: {e}")
            return False
    
    def run_all_validations(self) -> Dict:
        """Run all validation checks."""
        print("="*60)
        print("BOP Dataset Validation")
        print("="*60)
        
        self.validate_structure()
        self.validate_annotations()
        self.validate_images()
        self.validate_model()
        
        print("\n" + "="*60)
        print("Validation Summary")
        print("="*60)
        
        all_valid = all(self.results.values())
        
        for check, valid in tqdm(self.results.items()):
            status = "✓ PASS" if valid else "✗ FAIL"
            print(f"{check}: {status}")
        
        if all_valid:
            print("\n✓ All validations passed!")
        else:
            print("\n✗ Some validations failed. Fix issues above.")
        
        return self.results


class BOPDatasetVisualizer:
    """Visualize dataset frames with poses."""
    
    def __init__(self, dataset_dir, split='train'):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        
        # Load data
        with open(self.dataset_dir / f"{self.split}/{self.obj_id}/scene_gt.json") as f:
            self.scene_gt = json.load(f)
        
        # Load model
        model_path = self.dataset_dir / "models/obj_000001.ply"
        self.mesh = trimesh.load(str(model_path))
    
    def get_frame_data(self, frame_id: int) -> Dict:
        """Load RGB, depth, and pose for a frame."""
        rgb_path = self.dataset_dir / f"{self.split}/{self.obj_id}/rgb/{frame_id:06d}.png"
        depth_path = self.dataset_dir / f"{self.split}/{self.obj_id}/depth/{frame_id:06d}.png"
        
        rgb = np.array(Image.open(rgb_path))
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        
        # Ground truth pose
        gt = self.scene_gt[frame_id][0]
        R_m2c = np.array(gt['cam_R_m2c']).reshape(3, 3)
        t_m2c = np.array(gt['cam_t_m2c']).reshape(3, 1)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'R_m2c': R_m2c,
            't_m2c': t_m2c,
        }
    
    def visualize_frame(self, frame_id: int = 0, show_depth: bool = True):
        """Display frame with metadata."""
        data = self.get_frame_data(frame_id)
        
        # Create figure with subplots
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 5))
        
        # RGB image
        ax1 = fig.add_subplot(131)
        ax1.imshow(data['rgb'])
        ax1.set_title(f"RGB Frame {frame_id}")
        ax1.axis('off')
        
        # Depth visualization
        if show_depth:
            ax2 = fig.add_subplot(132)
            depth_vis = np.clip(data['depth'] / data['depth'].max(), 0, 1)
            ax2.imshow(depth_vis, cmap='gray')
            ax2.set_title(f"Depth (max: {data['depth'].max():.0f} mm)")
            ax2.axis('off')
        
        # Pose info
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        R = data['R_m2c']
        t = data['t_m2c'].flatten()
        
        # Convert rotation to axis-angle for visualization
        angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
        
        info_text = (
            f"Frame ID: {frame_id}\n"
            f"\nTranslation (mm):\n"
            f"  tx: {t[0]:8.2f}\n"
            f"  ty: {t[1]:8.2f}\n"
            f"  tz: {t[2]:8.2f}\n"
            f"\nRotation:\n"
            f"  Angle: {angle:.2f}°\n"
            f"  det(R): {np.linalg.det(R):.4f}\n"
            f"\nImage size:\n"
            f"  {data['rgb'].shape[1]}×{data['rgb'].shape[0]}\n"
            f"\nDepth range:\n"
            f"  {data['depth'].min():.0f}-{data['depth'].max():.0f} mm"
        )
        
        ax3.text(0.1, 0.5, info_text, fontfamily='monospace',
                fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_multiple(self, frame_ids: List[int] = None, num_frames: int = 5):
        """Show multiple frames in a grid."""
        import matplotlib.pyplot as plt
        
        if frame_ids is None:
            total_frames = len(self.scene_gt)
            frame_ids = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        fig, axes = plt.subplots(2, len(frame_ids), figsize=(4*len(frame_ids), 8))
        
        if len(frame_ids) == 1:
            axes = axes.reshape(2, 1)
        
        for col, frame_id in enumerate(frame_ids):
            data = self.get_frame_data(frame_id)
            
            # RGB
            axes[0, col].imshow(data['rgb'])
            axes[0, col].set_title(f"Frame {frame_id}")
            axes[0, col].axis('off')
            
            # Depth
            depth_vis = np.clip(data['depth'] / data['depth'].max(), 0, 1)
            axes[1, col].imshow(depth_vis, cmap='viridis')
            axes[1, col].set_title(f"Depth")
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.show()
def get_args():
    parser = argparse.ArgumentParser(
        description="validate a BOP-style dataset"
    )

    # Paths / IO
    parser.add_argument("--dataset-folder",type=str,default="path/to/your/model.ply",help="Folder to the dataset youre trying to validate",)
    parser.add_argument("--split",type=str,default="train",help="Which split to validate",)
    return parser.parse_args()

def main(args):
    """Example validation and visualization."""
    dataset_dir = args.dataset_folder
    
    # Validate dataset
    validator = BOPDatasetValidator(dataset_dir, split=args.split)
    results = validator.run_all_validations()
    
    # If valid, visualize some frames
    if all(results.values()):
        print("\n" + "="*60)
        print("Visualizing Dataset Frames")
        print("="*60)
        
        visualizer = BOPDatasetVisualizer(dataset_dir, split=args.split)
        
        # Show first frame detailed
        visualizer.visualize_frame(frame_id=0)
        
        # Show multiple frames
        visualizer.visualize_multiple(num_frames=6)


if __name__ == "__main__":
    """
    python ./bop_validate.py --dataset-folder ./data/BOP/issi/ --split train
    """
    args = get_args()
    main(args)
