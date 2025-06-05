import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import time
import json

class KITTILiDARProcessor:
    def __init__(self, data_dir, output_dir="output_analysis"):
        """
        Initialize the KITTI LiDAR processor
        
        Args:
            data_dir (str): Path to KITTI velodyne directory
            output_dir (str): Directory to save analysis results
        """
        self.KITTI_LIDAR_DIR = data_dir
        self.OUTPUT_DIR = output_dir
        self.REFERENCE_MAP_PATH = os.path.join(output_dir, "reference_map.pcd")
        
        # Processing parameters
        self.params = {
            'voxel_size': 0.1,               # Downsampling voxel size (meters)
            'icp_threshold': 0.2,            # ICP matching threshold
            'feature_radius': 0.3,           # Feature extraction radius
            'dbscan_eps': 0.5,              # DBSCAN clustering epsilon
            'dbscan_min_samples': 10,        # DBSCAN minimum samples
            'path_resolution': 0.5,          # Path planning resolution
            'safety_margin': 1.0,            # Obstacle avoidance margin
            'max_scans': 20                  # Maximum scans to process
        }
        
        # Create output directory
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Initialize reference map
        self.reference_map = self._initialize_reference_map()
    
    def _initialize_reference_map(self):
        """Initialize or load existing reference map"""
        if os.path.exists(self.REFERENCE_MAP_PATH):
            print("Loading existing reference map...")
            return o3d.io.read_point_cloud(self.REFERENCE_MAP_PATH)
        print("No reference map found - will create from first valid scan")
        return None
    
    def load_kitti_scan(self, bin_path):
        """
        Robust loading of KITTI .bin files with error handling
        
        Args:
            bin_path (str): Path to .bin file
            
        Returns:
            o3d.geometry.PointCloud: Processed point cloud
        """
        try:
            # Read binary file
            data = np.fromfile(bin_path, dtype=np.float32)
            
            # Verify data size
            if data.size % 4 != 0:
                print(f"Warning: File {os.path.basename(bin_path)} has {data.size} values (not divisible by 4)")
                data = data[:4*(data.size//4)]  # Truncate to nearest multiple of 4
            
            points = data.reshape(-1, 4)  # Reshape to Nx4 (x,y,z,intensity)
            
            # Validate points
            if points.shape[0] == 0:
                raise ValueError("Empty point cloud")
                
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            # Add intensity as color if available
            if points.shape[1] >= 4:
                intensities = points[:, 3]
                intensities = np.nan_to_num(intensities)  # Handle NaN/inf
                if np.max(intensities) > 0:  # Only normalize if non-zero
                    colors = np.zeros((points.shape[0], 3))
                    colors[:, :] = intensities.reshape(-1, 1) / np.max(intensities)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Downsample
            return pcd.voxel_down_sample(voxel_size=self.params['voxel_size'])
            
        except Exception as e:
            print(f"Error loading {os.path.basename(bin_path)}: {str(e)}")
            return o3d.geometry.PointCloud()  # Return empty point cloud

    def extract_features(self, pcd):
        """Extract FPFH features for point cloud registration"""
        # Estimate normals
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.params['voxel_size'] * 2,
                max_nn=30))
        
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.params['feature_radius'],
                max_nn=100))
        return fpfh

    def register_scan(self, scan_pcd):
        """Align current scan to reference map"""
        if self.reference_map is None:
            # First valid scan becomes reference
            self.reference_map = scan_pcd
            return np.identity(4), 1.0, "Initial reference created"
        
        # Feature extraction
        source_fpfh = self.extract_features(scan_pcd)
        target_fpfh = self.extract_features(self.reference_map)
        
        # Global registration
        distance_threshold = self.params['voxel_size'] * 1.5
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            scan_pcd, self.reference_map, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        
        # Local refinement with ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            scan_pcd, self.reference_map, self.params['icp_threshold'],
            result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        return icp_result.transformation, icp_result.fitness, "Registration successful"

    def update_reference_map(self, scan_pcd, transformation):
        """Update reference map with new scan"""
        scan_pcd.transform(transformation)
        self.reference_map += scan_pcd
        self.reference_map = self.reference_map.voxel_down_sample(
            voxel_size=self.params['voxel_size'])
        o3d.io.write_point_cloud(self.REFERENCE_MAP_PATH, self.reference_map)

    def detect_obstacles(self, pcd):
        """Detect obstacles using DBSCAN clustering"""
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return []
        
        # Cluster points
        db = DBSCAN(eps=self.params['dbscan_eps'],
                   min_samples=self.params['dbscan_min_samples']).fit(points)
        labels = db.labels_
        
        # Extract clusters (ignore noise labeled as -1)
        obstacles = []
        for cluster_id in set(labels) - {-1}:
            cluster_mask = (labels == cluster_id)
            cluster_points = points[cluster_mask]
            
            # Calculate bounding box
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            center = (min_bounds + max_bounds) / 2
            extent = max_bounds - min_bounds
            
            obstacles.append({
                'points': cluster_points,
                'center': center,
                'extent': extent,
                'label': cluster_id
            })
        
        return obstacles

    def plan_path(self, start, goal, obstacles):
        """Simple 3D A* path planning with obstacle avoidance"""
        if not obstacles:
            return np.array([start, goal])  # Straight line if no obstacles
        
        # Create obstacle KDTree
        obstacle_points = np.vstack([obs['points'] for obs in obstacles])
        obstacle_tree = KDTree(obstacle_points)
        
        # A* implementation
        open_set = {tuple(start)}
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): np.linalg.norm(np.array(start) - np.array(goal))}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            # Check if reached goal
            if np.linalg.norm(np.array(current) - np.array(goal)) < self.params['path_resolution']:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return np.array(path[::-1])  # Return reversed path
            
            open_set.remove(current)
            
            # Generate neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                            
                        neighbor = (
                            current[0] + dx * self.params['path_resolution'],
                            current[1] + dy * self.params['path_resolution'],
                            current[2] + dz * self.params['path_resolution'])
                        
                        # Check obstacle collision
                        dist, _ = obstacle_tree.query([neighbor], k=1)
                        if dist[0] < self.params['safety_margin']:
                            continue
                            
                        # Update scores
                        tentative_g = g_score[current] + np.linalg.norm(
                            np.array(neighbor) - np.array(current))
                        
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + np.linalg.norm(
                                np.array(neighbor) - np.array(goal))
                            if neighbor not in open_set:
                                open_set.add(neighbor)
        
        return np.array([start, goal])  # Fallback straight path

    def visualize(self, scan_pcd, obstacles=None, path=None, transformation=None):
        """Visualize point clouds and processing results"""
        geometries = []
        
        # Add reference map if available
        if self.reference_map:
            ref_map = self.reference_map.voxel_down_sample(
                voxel_size=self.params['voxel_size'])
            ref_map.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
            geometries.append(ref_map)
        
        # Add current scan
        if transformation is not None:
            scan_pcd.transform(transformation)
        scan_pcd.paint_uniform_color([0, 0, 1])  # Blue
        geometries.append(scan_pcd)
        
        # Add obstacles (red)
        if obstacles:
            for obs in obstacles:
                obs_pcd = o3d.geometry.PointCloud()
                obs_pcd.points = o3d.utility.Vector3dVector(obs['points'])
                obs_pcd.paint_uniform_color([1, 0, 0])
                geometries.append(obs_pcd)
                
                # Add bounding box
                bbox = o3d.geometry.OrientedBoundingBox(
                    center=obs['center'],
                    extent=obs['extent'])
                bbox.color = [1, 0, 0]
                geometries.append(bbox)
        
        # Add path (green)
        if path is not None and len(path) > 1:
            lines = [[i, i+1] for i in range(len(path)-1)]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path),
                lines=o3d.utility.Vector2iVector(lines))
            line_set.paint_uniform_color([0, 1, 0])
            geometries.append(line_set)
        
        # Add coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geometries.append(frame)
        
        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name="LiDAR Processing Results",
            width=1280,
            height=720)

    def save_results(self, scan_name, obstacles, path, reg_quality):
        """Save processing results to JSON file"""
        result = {
            'scan': scan_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'num_points': len(obstacles[0]['points']) if obstacles else 0,
            'num_obstacles': len(obstacles),
            'registration_quality': reg_quality,
            'path_length': float(np.sum(np.linalg.norm(
                path[1:] - path[:-1], axis=1))) if path is not None else 0,
            'parameters': self.params
        }
        
        output_path = os.path.join(
            self.OUTPUT_DIR,
            f"{os.path.splitext(scan_name)[0]}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

    def process_directory(self):
        """Process all KITTI scans in directory"""
        bin_files = sorted([
            f for f in os.listdir(self.KITTI_LIDAR_DIR) 
            if f.endswith('.bin') and os.path.getsize(
                os.path.join(self.KITTI_LIDAR_DIR, f)) > 0])
        
        if not bin_files:
            print(f"No valid .bin files found in {self.KITTI_LIDAR_DIR}")
            return
        
        print(f"\nFound {len(bin_files)} scans. Processing first {self.params['max_scans']}...")
        
        for bin_file in tqdm(bin_files[:self.params['max_scans']]):
            bin_path = os.path.join(self.KITTI_LIDAR_DIR, bin_file)
            
            try:
                # 1. Load scan
                scan_pcd = self.load_kitti_scan(bin_path)
                if len(scan_pcd.points) == 0:
                    continue
                
                # 2. Register to reference
                transform, reg_quality, _ = self.register_scan(scan_pcd)
                print(f"{bin_file}: Reg quality = {reg_quality:.3f}")
                
                # 3. Update reference
                self.update_reference_map(scan_pcd, transform)
                
                # 4. Obstacle detection
                obstacles = self.detect_obstacles(scan_pcd)
                
                # 5. Path planning (example: bottom to top of scan)
                points = np.asarray(scan_pcd.points)
                min_bounds = np.min(points, axis=0)
                max_bounds = np.max(points, axis=0)
                start = min_bounds + [0, 0, 1]  # 1m above min
                goal = max_bounds - [0, 0, 1]   # 1m below max
                path = self.plan_path(start, goal, obstacles)
                
                # 6. Visualization
                self.visualize(scan_pcd, obstacles, path, transform)
                
                # 7. Save results
                self.save_results(bin_file, obstacles, path, reg_quality)
                
            except Exception as e:
                print(f"\nError processing {bin_file}: {str(e)}")
                continue

if __name__ == "__main__":
    # Update this path to your KITTI velodyne directory
    DATA_DIR = r"E:\All the document in desktop\Pipe Line new papaer\data\kitti\velodyne\training\velodyne"
    
    processor = KITTILiDARProcessor(DATA_DIR)
    processor.process_directory()