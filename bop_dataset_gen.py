"""
BOP Dataset Generation Script for KeyGNET Training

Creates a BOP-style dataset from a single PLY mesh file with restricted viewpoints.
Renders from front-facing hemisphere only (±30° from front).

Requirements:
    - trimesh
    - numpy
    - opencv-python (cv2)
    - pyrender
    - imageio
    - json
    - PIL
"""

import os
import json
import numpy as np
from pathlib import Path
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Optional but recommended imports
try:
    import trimesh
    import pyrender
    import cv2
    from PIL import Image
    import imageio
except ImportError as e:
    print(f"Warning: Missing dependency {e}. Install with: pip install trimesh pyrender opencv-python imageio pillow")


class BOPDatasetGenerator:
    """Generate BOP-format dataset from a single PLY mesh with constrained viewpoints."""
    
    def __init__(self, args, split='train', obj_id=1):
        """
        Initialize dataset generator.
        
        Args:
            ply_path: Path to input PLY file
            output_dir: Output directory for BOP dataset
            obj_id: Object ID in BOP format (default 1)
            cam_distance_mm: Distance of camera from object center in mm
        """
        self.args = args
        self.split=split
        self.ply_path = Path(args.ply_file)
        self.output_dir = Path(args.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.obj_id = obj_id
        self.cam_distance_mm = args.camera_distance_mm
        
        # Load mesh
        print(f"Loading mesh from {self.ply_path}...")
        self.mesh = trimesh.load(str(self.ply_path))
        
        # Normalize mesh to origin and scale
        self._normalize_mesh()
        
        # Create output structure
        self._setup_directory_structure()
        
        # Camera intrinsics (RGB-D camera like RealSense)
        self.cam_K = np.array([
            [args.cam_fx, 0, args.cam_cx],
            [0, args.cam_fy, args.cam_cy],
            [0, 0, 1]
        ], dtype=np.float32)  # For 640x480 image
        
        self.img_width = args.img_width
        self.img_height = args.img_height
        self.depth_scale = 1.0  # mm
        
        print(f"Mesh loaded. Bounds: {self.mesh.bounds}")
        print(f"Mesh diameter: {self.mesh.extents.max():.2f} mm")
        
    def _normalize_mesh(self):
        """Center mesh at origin and ensure units are reasonable."""
        # Center at origin
        center = self.mesh.centroid
        self.mesh.vertices -= center
        
        # Get bounds in mm
        bounds = self.mesh.bounds
        mesh_extent = np.linalg.norm(bounds[1] - bounds[0])
        
        print(f"Mesh extent: {mesh_extent:.2f} mm")
        
    def _setup_directory_structure(self):
        """Create BOP dataset directory structure."""
        # Main directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.train_dir = self.output_dir / self.split / f"{self.obj_id}"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for train split
        (self.train_dir / "rgb").mkdir(exist_ok=True)
        (self.train_dir / "depth").mkdir(exist_ok=True)
        (self.train_dir / "mask").mkdir(exist_ok=True)
        (self.train_dir / "mask_visib").mkdir(exist_ok=True)
        
        print(f"Created directory structure in {self.output_dir}")
        
    def _generate_front_hemisphere_viewpoints(self, num_viewpoints=50, 
                                              front_angle_range=0.0):
        """
        Generate camera poses from front-facing hemisphere with angle constraint.
        
        Args:
            num_viewpoints: Number of viewpoints to generate
            front_angle_range: Max angle from front view (in degrees)
            
        Returns:
            List of (azimuth, elevation) angles in degrees
        """
        viewpoints = []
        
        # Generate viewpoints in front hemisphere
        for i in range(num_viewpoints):
            # Random azimuth around front (0°)
            azimuth = np.random.uniform(0, 180)
            
            # Random elevation in upper hemisphere
            # Elevation: 0° = horizontal, 90° = looking down
            elevation = np.random.uniform(0+front_angle_range,180-front_angle_range)
            
            viewpoints.append((azimuth, elevation))
        
        return viewpoints
    
    def _azimuth_elevation_to_pose(self, azimuth_deg, elevation_deg, 
                                   distance_mm):
        """
        Convert spherical coordinates to camera pose.
        
        Convention:
        - Azimuth: 0° = +X axis (front), 90° = +Y axis (left), -90° = -Y axis (right)
        - Elevation: 0° = horizontal plane, 90° = looking down (+Z)
        - Distance: distance from origin in mm
        
        Returns:
            (R_m2c, t_m2c): Model-to-camera rotation and translation
        """
        azim_rad = np.deg2rad(azimuth_deg)
        elev_rad = np.deg2rad(elevation_deg)
        
        # Spherical to Cartesian
        x = distance_mm * np.cos(elev_rad) * np.cos(azim_rad)
        y = distance_mm * np.cos(elev_rad) * np.sin(azim_rad)
        z = distance_mm * np.sin(elev_rad)
        
        cam_pos = np.array([x, y, z])
        
        # Camera looks at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        
        # Up vector (Z-axis in world)
        up = np.array([0, 0, 1])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recalculate up (ensure orthogonality)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Rotation matrix (world to camera)
        R_w2c = np.stack([right, up, -forward], axis=0)
        
        # Translation (world to camera)
        t_w2c = -R_w2c @ cam_pos
        
        # For BOP format, we need object-to-camera pose
        # Assuming object is at origin in world frame:
        R_m2c = R_w2c
        t_m2c = t_w2c
        
        return R_m2c, t_m2c, cam_pos
    
    def _render_frame(self, R_m2c, t_m2c, frame_id):
        """
        Render RGB, depth, and masks for given camera pose.
        
        Args:
            R_m2c: 3x3 rotation matrix (model to camera)
            t_m2c: 3x1 translation vector (model to camera)
            frame_id: Frame ID for output files
            
        Returns:
            Tuple of (rgb, depth, mask, mask_visib)
        """
        # Create Pyrender scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])
        
        # Create mesh with material
        mesh_pyrender = pyrender.Mesh.from_trimesh(self.mesh, smooth=True)
        scene.add(mesh_pyrender)
        # --- Add lights ---
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 0, 1.5 * self.cam_distance_mm / 1000.0])
        scene.add(light, pose=light_pose)

        light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = np.array([-1, -1, 1])
        scene.add(light2, pose=light2_pose)
        # --- end lights ---
        # Create camera pose in world frame
        # Convert m2c to camera node transform
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = R_m2c
        pose_w2c[:3, 3] = t_m2c.flatten()
        
        # Camera matrix
        camera = pyrender.IntrinsicsCamera(
            fx=self.cam_K[0, 0],
            fy=self.cam_K[1, 1],
            cx=self.cam_K[0, 2],
            cy=self.cam_K[1, 2],
            znear=10.0,
            zfar=5000.0
        )
        
        scene.add(camera, pose=np.linalg.inv(pose_w2c))
        
        # Render
        r = pyrender.OffscreenRenderer(self.img_width, self.img_height)
        
        # Render RGB
        rgb, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        
        # Convert to 8-bit RGB
        rgb = (rgb[:, :, :3] * 255).astype(np.uint8)
        
        # Convert depth to mm (uint16 format for BOP)
        depth_mm = (depth * self.depth_scale).astype(np.uint16)
        
        # Generate mask (non-zero depth pixels)
        mask = (depth > 0).astype(np.uint8) * 255
        mask_visib = mask.copy()  # All visible for now
        
        return rgb, depth_mm, mask, mask_visib
    
    def _save_frame_data(self, rgb, depth, mask, mask_visib, frame_id, 
                        R_m2c, t_m2c):
        """Save rendered frame data and update annotations."""
        # Save RGB image
        rgb_path = self.train_dir / "rgb" / f"{frame_id:06d}.png"
        Image.fromarray(rgb).save(str(rgb_path))
        
        # Save depth image
        depth_path = self.train_dir / "depth" / f"{frame_id:06d}.png"
        Image.fromarray(depth).save(str(depth_path))
        
        # Save masks
        mask_path = self.train_dir / "mask" / f"{frame_id:06d}_{0:06d}.png"
        Image.fromarray(mask).save(str(mask_path))
        
        mask_visib_path = self.train_dir / "mask_visib" / f"{frame_id:06d}_{0:06d}.png"
        Image.fromarray(mask_visib).save(str(mask_visib_path))
        
        # Prepare ground truth annotation
        gt_annotation = {
            "cam_R_m2c": R_m2c.flatten().tolist(),
            "cam_t_m2c": t_m2c.flatten().tolist(),
            "obj_id": self.obj_id,
            "cam_K": self.cam_K.flatten().tolist(),
        }
        
        return gt_annotation
    
    def generate_dataset(self, num_frames=100):
        """
        Generate full BOP dataset with specified number of frames.
        
        Args:
            num_frames: Number of training frames to render
        """
        print(f"\nGenerating {num_frames} training frames...")
        
        gt_annotations = []
        scene_camera_data = {}
        
        # Generate viewpoints
        viewpoints = self._generate_front_hemisphere_viewpoints(num_frames, self.args.angle_range_deg)
        
        for frame_id, (azim, elev) in tqdm(enumerate(viewpoints)):
            # print(f"  Rendering frame {frame_id + 1}/{num_frames} "
            #         f"(azimuth={azim:.1f}°, elevation={elev:.1f}°)...")
            
            # Generate pose
            R_m2c, t_m2c, cam_pos = self._azimuth_elevation_to_pose(
                azim, elev, self.cam_distance_mm
            )
            
            # Render
            rgb, depth, mask, mask_visib = self._render_frame(R_m2c, t_m2c, frame_id)
            
            # Save and get annotations
            gt_anno = self._save_frame_data(rgb, depth, mask, mask_visib, 
                                            frame_id, R_m2c, t_m2c)
            gt_annotations.append([gt_anno])
            
            # Camera info for this frame
            scene_camera_data[frame_id] = {
                "cam_K": self.cam_K.tolist(),
                "depth_scale": self.depth_scale,
            }
        
        # Save annotations
        self._save_annotations(gt_annotations, scene_camera_data)
        
        # Save dataset metadata
        self._save_dataset_info(num_frames)
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Total frames: {num_frames}")
        print(f"  - Models: {self.models_dir}")
        print(f"  - Training data: {self.train_dir}")
        
    def _save_annotations(self, gt_annotations, scene_camera_data):
        """Save ground truth and camera annotations in BOP format."""
        # Ground truth annotations
        gt_path = self.train_dir / "scene_gt.json"
        with open(gt_path, 'w') as f:
            json.dump(gt_annotations, f, indent=2)
        
        # Camera parameters
        scene_camera_path = self.train_dir / "scene_camera.json"
        with open(scene_camera_path, 'w') as f:
            json.dump(scene_camera_data, f, indent=2)
        
        print(f"  Saved annotations to {gt_path}")
        print(f"  Saved camera info to {scene_camera_path}")
        
    def _save_dataset_info(self, num_frames):
        """Save dataset-level metadata files."""
        # Camera intrinsics (global)
        camera_data = {
            "1": {
                "cam_K": self.cam_K.tolist(),
                "depth_scale": self.depth_scale,
                "width": self.img_width,
                "height": self.img_height,
            }
        }
        
        camera_path = self.output_dir / "camera.json"
        with open(camera_path, 'w') as f:
            json.dump(camera_data, f, indent=2)
        
        # Model info
        model_bounds = self.mesh.bounds
        model_extent = np.linalg.norm(model_bounds[1] - model_bounds[0])
        
        models_info = {
            str(self.obj_id): {
                "diameter": float(model_extent),
                "min_x": float(model_bounds[0, 0]),
                "min_y": float(model_bounds[0, 1]),
                "min_z": float(model_bounds[0, 2]),
                "max_x": float(model_bounds[1, 0]),
                "max_y": float(model_bounds[1, 1]),
                "max_z": float(model_bounds[1, 2]),
                "size_x": float(model_bounds[1, 0] - model_bounds[0, 0]),
                "size_y": float(model_bounds[1, 1] - model_bounds[0, 1]),
                "size_z": float(model_bounds[1, 2] - model_bounds[0, 2]),
            }
        }
        
        models_info_path = self.models_dir / "models_info.json"
        with open(models_info_path, 'w') as f:
            json.dump(models_info, f, indent=2)
        
        # Dataset info
        dataset_info = {
            "name": self.ply_path.stem,
            "description": f"BOP dataset generated from {self.ply_path.name} "
                          f"with front-facing viewpoints",
            "object_count": 1,
            "train_count": num_frames,
            "val_count": 0,
            "test_count": 0,
            "train_split": "front_hemisphere_only",
        }
        
        dataset_info_path = self.output_dir / "dataset_info.json"
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Copy model to models directory
        model_out_path = self.models_dir / f"obj_{self.obj_id:06d}.ply"
        self.mesh.export(str(model_out_path))
        
        print(f"  Saved dataset metadata")
        print(f"  Copied model to {model_out_path}")

def get_args():
    parser = argparse.ArgumentParser(
        description="Render BOP-style dataset from a PLY 3D model."
    )

    # Paths / IO
    parser.add_argument("--ply-file",type=str,default="path/to/your/model.ply",help="Path to the input PLY model file.",)
    parser.add_argument("--output-dir",type=str,default="./bop_dataset",help="Output directory for the generated BOP dataset.",)

    # Dataset / rendering config
    parser.add_argument("--num-frames",type=int,default=100,help="Number of frames to render.",)
    parser.add_argument("--camera-distance-mm",type=float,default=800.0,help="Camera distance from object in millimeters.",)
    parser.add_argument("--angle-range-deg",type=float,default=30.0,help="Half-angle (in degrees) for viewpoints around the front.",)

    # Image resolution
    parser.add_argument("--img-width",type=int,default=640,help="Output image width in pixels.",)
    parser.add_argument("--img-height",type=int,default=480,help="Output image height in pixels.",)

    # Camera intrinsics (fx, fy, cx, cy) for RealSense
    parser.add_argument("--cam-fx",type=float,default=1075.65,help="Camera focal length fx in pixels.",)
    parser.add_argument("--cam-fy",type=float,default=1073.90,help="Camera focal length fy in pixels.",)
    parser.add_argument("--cam-cx",type=float,default=320,help="Principal point x coordinate (cx) in pixels.",)
    parser.add_argument("--cam-cy",type=float,default=240,help="Principal point y coordinate (cy) in pixels.",)

    return parser.parse_args()

def main(args):
    """Example usage of BOPDatasetGenerator."""
    
    # Configuration
    PLY_FILE = args.ply_file
    OUTPUT_DIR = args.output_dir
    NUM_FRAMES = args.num_frames
    
    # Check file exists
    if not Path(PLY_FILE).exists():
        print(f"Error: PLY file not found at {PLY_FILE}")
        print("Please edit PLY_FILE variable with the correct path.")
        return
    
    # e.g. 80/20 split
    num_train = int(0.8 * NUM_FRAMES)
    num_test  = NUM_FRAMES - num_train

    # TRAIN
    gen_train = BOPDatasetGenerator(args, obj_id=1, split="train")
    gen_train.generate_dataset(num_frames=num_train)

    # TEST
    gen_test = BOPDatasetGenerator(args, obj_id=1, split="test")
    gen_test.generate_dataset(num_frames=num_test)
    
    print("\n" + "="*60)
    print("BOP Dataset Created Successfully!")
    print("="*60)
    print(f"\nDirectory structure:")
    print(f"{OUTPUT_DIR}/")
    print(f"├── camera.json")
    print(f"├── dataset_info.json")
    print(f"├── models/")
    print(f"│   ├── models_info.json")
    print(f"│   └── obj_000001.ply")
    print(f"└── train/1/")
    print(f"    ├── rgb/          (*.png)")
    print(f"    ├── depth/        (*.png)")
    print(f"    ├── mask/         (*.png)")
    print(f"    ├── mask_visib/   (*.png)")
    print(f"    ├── scene_gt.json")
    print(f"    └── scene_camera.json")
    print("\nYou can now use this dataset with KeyGNET!")


if __name__ == "__main__":
    """
    python ./bop_dataset_gen.py --ply-file ./object_files/active_interface.ply --output-dir ./data/BOP/issi --num-frames 1000 --angle-range-deg 30
    """
    args = get_args()
    main(args)
