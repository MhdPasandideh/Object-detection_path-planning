import open3d as o3d
import numpy as np
import os
from tqdm import tqdm  # for progress bar

# Constants
KITTI_LIDAR_DIR = r"E:\All the document in desktop\Pipe Line new papaer\data\kitti\velodyne\training\velodyne"

def read_kitti_bin_file(bin_path):
    """Read KITTI .bin file and return Open3D point cloud"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Add intensity as color (normalized to 0-1)
    if points.shape[1] >= 4:
        intensities = points[:, 3]
        # Normalize and create RGB colors (grayscale)
        colors = np.zeros((points.shape[0], 3))
        colors[:, :] = intensities.reshape(-1, 1) / intensities.max()
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_lidar(pcd, window_name="KITTI LiDAR Data"):
    """Visualize point cloud with Open3D"""
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # Set view controls
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])  # Looking along the negative Z-axis
    view_ctl.set_up([0, -1, 0])     # Up is Y-axis
    
    # Set rendering options
    render_opt = vis.get_render_option()
    render_opt.point_size = 1.5
    render_opt.background_color = np.asarray([0.05, 0.05, 0.05])
    render_opt.show_coordinate_frame = True
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def process_kitti_directory(directory):
    """Process all .bin files in a KITTI directory"""
    # Get all .bin files in directory
    bin_files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    bin_files.sort()  # Sort numerically
    
    if not bin_files:
        print(f"No .bin files found in directory: {directory}")
        return
    
    print(f"Found {len(bin_files)} LiDAR scans in directory")
    
    # Visualize each file one by one
    for i, bin_file in enumerate(tqdm(bin_files, desc="Processing LiDAR scans")):
        bin_path = os.path.join(directory, bin_file)
        
        try:
            # Read point cloud
            pcd = read_kitti_bin_file(bin_path)
            
            # Print basic info
            print(f"\nScan {i+1}/{len(bin_files)}: {bin_file}")
            print(f"Number of points: {len(pcd.points)}")
            
            # Visualize
            visualize_lidar(pcd, window_name=f"KITTI LiDAR: {bin_file}")
            
        except Exception as e:
            print(f"\nError processing {bin_file}: {str(e)}")
            continue

if __name__ == "__main__":
    # Verify directory exists
    if not os.path.exists(KITTI_LIDAR_DIR):
        print(f"Directory not found: {KITTI_LIDAR_DIR}")
    else:
        process_kitti_directory(KITTI_LIDAR_DIR)