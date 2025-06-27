import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PathPlanner:
    """Class for path planning based on object detections and vehicle state."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detections = self.load_detections(config['detections_path'])
        self.oxts_data = self.load_oxts(config['oxts_dir'])
        logger.info("Path planner initialized")
    
    def load_detections(self, detections_path: str) -> pd.DataFrame:
        """Load object detections from CSV."""
        if not os.path.exists(detections_path):
            raise FileNotFoundError(f"Detections file {detections_path} not found")
        
        df = pd.read_csv(detections_path)
        if df.empty:
            logger.warning("No detections found in the input file")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def load_oxts(self, oxts_dir: str) -> pd.DataFrame:
        """Load OXTS data from directory."""
        if not os.path.exists(oxts_dir):
            raise FileNotFoundError(f"OXTS directory {oxts_dir} not found")
        
        # Find all OXTS files
        oxts_files = glob.glob(os.path.join(oxts_dir, '*.txt'))
        if not oxts_files:
            raise ValueError(f"No OXTS files found in {oxts_dir}")
        
        # Load and concatenate all OXTS data
        all_data = []
        for file in oxts_files:
            # Read space-separated values
            data = np.loadtxt(file)
            if data.size > 0:
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                all_data.append(data)
        
        if not all_data:
            raise ValueError("No valid OXTS data found")
        
        # Create DataFrame with appropriate columns
        columns = [
            'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 
            'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 
            'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 
            'pos_accuracy', 'vel_accuracy', 'navstat', 'numsats', 
            'posmode', 'velmode', 'orimode'
        ]
        
        df = pd.DataFrame(np.vstack(all_data), columns=columns)
        
        # Load corresponding timestamps
        timestamp_file = os.path.join(oxts_dir, 'timestamps.txt')
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                timestamps = [line.strip() for line in f if line.strip()]
            
            # Match timestamps with data
            if len(timestamps) == len(df):
                df['timestamp'] = pd.to_datetime(timestamps)
            else:
                logger.warning("Mismatch between OXTS data and timestamps")
        
        return df
    
    def match_detections_to_oxts(self):
        """Match detections with closest OXTS data by timestamp."""
        if self.detections.empty or self.oxts_data.empty:
            return pd.DataFrame()
        
        # Convert timestamps to numeric for matching
        detections_ts = self.detections['timestamp'].astype('int64') // 10**9
        oxts_ts = self.oxts_data['timestamp'].astype('int64') // 10**9
        
        # Find closest OXTS entry for each detection
        matched_indices = []
        for ts in detections_ts:
            idx = (oxts_ts - ts).abs().idxmin()
            matched_indices.append(idx)
        
        # Join data
        matched_data = self.detections.copy()
        matched_oxts = self.oxts_data.iloc[matched_indices].reset_index(drop=True)
        matched_data = pd.concat([matched_data, matched_oxts], axis=1)
        
        return matched_data
    
    def plan_path(self):
        """Main path planning algorithm."""
        # Match detections with vehicle state
        combined_data = self.match_detections_to_oxts()
        if combined_data.empty:
            logger.error("No matched data for path planning")
            return None
        
        # Simple path planning based on detected objects and vehicle state
        paths = []
        for _, row in combined_data.iterrows():
            # Get current state
            x, y = row['lon'], row['lat']  # Using longitude/latitude as proxy for x/y
            yaw = row['yaw']
            speed = np.sqrt(row['vn']**2 + row['ve']**2)
            
            # Get nearby objects (simplified)
            nearby_objs = combined_data[
                (combined_data['timestamp'] - row['timestamp']).abs() < pd.Timedelta('1s')
            ]
            
            # Simple avoidance strategy
            if not nearby_objs.empty:
                # Calculate relative positions
                rel_pos = nearby_objs[['center_x', 'center_y']].values - np.array([x, y])
                
                # Find closest object in front
                front_mask = (rel_pos[:, 0] * np.cos(yaw) + rel_pos[:, 1] * np.sin(yaw)) > 0
                if front_mask.any():
                    closest_idx = np.argmin(np.linalg.norm(rel_pos[front_mask], axis=1))
                    closest_obj = rel_pos[front_mask][closest_idx]
                    
                    # Calculate avoidance path (simple offset)
                    avoid_offset = np.array([-closest_obj[1], closest_obj[0]])
                    avoid_offset = avoid_offset / np.linalg.norm(avoid_offset) * 5  # 5m offset
                    
                    path = {
                        'timestamp': row['timestamp'],
                        'start_pos': (x, y),
                        'end_pos': (x + 10*np.cos(yaw), y + 10*np.sin(yaw)),  # Default path
                        'avoid_pos': (x + avoid_offset[0], y + avoid_offset[1]),  # Avoidance point
                        'speed': speed,
                        'objects': len(nearby_objs)
                    }
                else:
                    # No objects in front, continue straight
                    path = {
                        'timestamp': row['timestamp'],
                        'start_pos': (x, y),
                        'end_pos': (x + 10*np.cos(yaw), y + 10*np.sin(yaw)),
                        'avoid_pos': None,
                        'speed': speed,
                        'objects': 0
                    }
                
                paths.append(path)
        
        return paths
    
    def visualize_path(self, paths: List[Dict]):
        """Visualize and save the planned path."""
        if not paths:
            logger.warning("No paths to visualize")
            return
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Plot path
        plt.figure(figsize=(10, 10))
        
        # Plot vehicle trajectory
        x_coords = [p['start_pos'][0] for p in paths]
        y_coords = [p['start_pos'][1] for p in paths]
        plt.plot(x_coords, y_coords, 'b-', label='Vehicle Path')
        
        # Plot avoidance points
        avoid_x = [p['avoid_pos'][0] for p in paths if p['avoid_pos'] is not None]
        avoid_y = [p['avoid_pos'][1] for p in paths if p['avoid_pos'] is not None]
        if avoid_x:
            plt.scatter(avoid_x, avoid_y, c='r', marker='x', label='Avoidance Points')
        
        # Plot start and end
        plt.scatter(x_coords[0], y_coords[0], c='g', marker='o', label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], c='r', marker='o', label='End')
        
        plt.xlabel('Longitude (approx x)')
        plt.ylabel('Latitude (approx y)')
        plt.title('Planned Path with Obstacle Avoidance')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_path = os.path.join(self.config['output_dir'], 'path_plan.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved path visualization to {output_path}")
        
        return output_path
    
    def save_path_data(self, paths: List[Dict]):
        """Save path planning results to CSV."""
        if not paths:
            logger.warning("No path data to save")
            return
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(paths)
        
        # Save to CSV
        output_csv = os.path.join(self.config['output_dir'], 'path_plan.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved path data to {output_csv}")
        
        return output_csv

def main():
    """Main function for path planning."""
    # Configuration
    config = {
        'detections_path': '/content/objectdetection/detections.csv',
        'oxts_dir': '/content/oxts',
        'output_dir': '/content/pathplanning'
    }
    
    try:
        # Initialize path planner
        planner = PathPlanner(config)
        
        # Run path planning
        paths = planner.plan_path()
        if not paths:
            raise ValueError("No paths generated")
        
        # Visualize and save results
        planner.visualize_path(paths)
        planner.save_path_data(paths)
        
    except Exception as e:
        logger.error(f"Error in path planning pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()