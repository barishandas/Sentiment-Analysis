"""
Simplified Implementation of AI-Enhanced Optical Tweezers Tracking and Analysis System

This code provides a streamlined framework for tracking and analyzing particles in 
optical tweezers experiments. It includes:
1. Basic data handling and preprocessing
2. Particle detection using traditional computer vision
3. Simple particle tracking
4. Basic physical analysis
5. Visualization tools

Requirements:
- numpy
- opencv-python
- pandas
- matplotlib
- scikit-learn
- scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Tuple, Optional, Union
import time


class OpticalTweezersData:
    """Class for handling optical tweezers data"""
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        
    def load_image_sequence(self, folder_path: str = None) -> List[np.ndarray]:
        """Load a sequence of images from a folder"""
        if folder_path is None and self.data_path is None:
            raise ValueError("No data path provided")
        
        path = folder_path if folder_path else self.data_path
        image_files = sorted([f for f in os.listdir(path) 
                             if f.endswith(('.tif', '.png', '.jpg'))])
        
        images = []
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        
        return images
    
    def generate_synthetic_data(self, n_frames: int = 100, 
                               n_particles: int = 5, 
                               frame_size: Tuple[int, int] = (512, 512),
                               noise_level: float = 0.1) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate synthetic data for testing"""
        h, w = frame_size
        images = []
        positions = np.zeros((n_frames, n_particles, 2))
        
        # Initialize particle positions
        init_positions = np.random.rand(n_particles, 2) * np.array([w, h])
        
        # Parameters
        trap_stiffness = 0.1
        particle_radius = 5
        diffusion_coeff = 0.5
        
        # Generate frames
        current_positions = init_positions.copy()
        for frame in range(n_frames):
            # Calculate forces (simple harmonic trap)
            force = -trap_stiffness * (current_positions - init_positions)
            
            # Update positions with random motion
            current_positions = (current_positions + force + 
                                diffusion_coeff * np.random.randn(n_particles, 2))
            
            # Ensure particles stay within bounds
            current_positions = np.clip(current_positions, 0, 
                                      [w-1, h-1])
            
            # Create empty image
            img = np.zeros((h, w), dtype=np.float32)
            
            # Draw particles as Gaussian spots
            for i, (x, y) in enumerate(current_positions):
                x_int, y_int = int(x), int(y)
                
                # Define region around particle
                y_min = max(0, y_int - particle_radius)
                y_max = min(h, y_int + particle_radius + 1)
                x_min = max(0, x_int - particle_radius)
                x_max = min(w, x_int + particle_radius + 1)
                
                # Create meshgrid for the region
                y_coords, x_coords = np.meshgrid(
                    np.arange(y_min, y_max),
                    np.arange(x_min, x_max),
                    indexing='ij'
                )
                
                # Calculate distance from particle center
                dist_squared = (x_coords - x)**2 + (y_coords - y)**2
                
                # Add Gaussian intensity
                intensity = np.exp(-dist_squared / (2 * (particle_radius/2)**2))
                img[y_min:y_max, x_min:x_max] = np.maximum(
                    img[y_min:y_max, x_min:x_max], 
                    intensity
                )
            
            # Add noise
            img += np.random.randn(h, w) * noise_level
            img = np.clip(img, 0, 1)
            
            # Convert to 8-bit
            img = (img * 255).astype(np.uint8)
            
            # Store results
            images.append(img)
            positions[frame] = current_positions
        
        return images, positions


class ParticleDetector:
    """Class for detecting particles in images using traditional CV methods"""
    def __init__(self, threshold: int = 128, 
                min_size: int = 10,
                max_size: int = 1000,
                blur_sigma: float = 1.0):
        self.threshold = threshold
        self.min_size = min_size
        self.max_size = max_size
        self.blur_sigma = blur_sigma
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection"""
        # Make a copy to avoid modifying original
        processed = image.copy()
        
        # Apply Gaussian blur
        processed = gaussian_filter(processed, sigma=self.blur_sigma)
        
        return processed
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect particles in an image"""
        # Preprocess
        processed = self.preprocess(image)
        
        # Threshold
        _, binary = cv2.threshold(processed, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_size <= area <= self.max_size:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append([cx, cy])
        
        return np.array(centroids)


class ParticleTracker:
    """Class for tracking particles across frames"""
    def __init__(self, max_distance: float = 30, max_frames_lost: int = 5):
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.tracks = []
        self.next_id = 0
    
    def track(self, centroids_list: List[np.ndarray]) -> List[Dict]:
        """Track particles across multiple frames"""
        self.tracks = []
        active_tracks = []
        
        # Process all frames
        for frame_idx, centroids in enumerate(centroids_list):
            # Skip empty frames
            if len(centroids) == 0:
                continue
            
            # First frame - initialize tracks
            if frame_idx == 0 or len(active_tracks) == 0:
                for centroid in centroids:
                    track = {
                        'id': self.next_id,
                        'positions': [centroid],
                        'frames': [frame_idx],
                        'last_seen': frame_idx
                    }
                    active_tracks.append(track)
                    self.tracks.append(track)
                    self.next_id += 1
                continue
            
            # Calculate distances between current detections and active tracks
            cost_matrix = np.zeros((len(active_tracks), len(centroids)))
            for i, track in enumerate(active_tracks):
                last_pos = track['positions'][-1]
                for j, pos in enumerate(centroids):
                    cost_matrix[i, j] = np.sqrt(np.sum((last_pos - pos)**2))
            
            # Assign detections to tracks using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Lists to track which detections and tracks have been assigned
            assigned_detections = set()
            updated_tracks = []
            
            # Update assigned tracks
            for track_idx, detection_idx in zip(row_indices, col_indices):
                # Only assign if distance is below threshold
                if cost_matrix[track_idx, detection_idx] <= self.max_distance:
                    track = active_tracks[track_idx]
                    track['positions'].append(centroids[detection_idx])
                    track['frames'].append(frame_idx)
                    track['last_seen'] = frame_idx
                    updated_tracks.append(track)
                    assigned_detections.add(detection_idx)
            
            # Add tracks that weren't updated but are still active
            for i, track in enumerate(active_tracks):
                if i not in row_indices and frame_idx - track['last_seen'] <= self.max_frames_lost:
                    updated_tracks.append(track)
            
            # Create new tracks for unassigned detections
            for j in range(len(centroids)):
                if j not in assigned_detections:
                    track = {
                        'id': self.next_id,
                        'positions': [centroids[j]],
                        'frames': [frame_idx],
                        'last_seen': frame_idx
                    }
                    updated_tracks.append(track)
                    self.tracks.append(track)
                    self.next_id += 1
            
            # Update active tracks
            active_tracks = updated_tracks
        
        # Convert position lists to numpy arrays for easier processing
        for track in self.tracks:
            track['positions'] = np.array(track['positions'])
            track['frames'] = np.array(track['frames'])
        
        return self.tracks


class PhysicalAnalyzer:
    """Class for physical analysis of particle trajectories"""
    def __init__(self, pixel_size: float = 0.1, frame_rate: float = 30):
        self.pixel_size = pixel_size  # µm per pixel
        self.frame_rate = frame_rate  # frames per second
        self.kT = 4.11e-21  # Thermal energy at room temp in Joules
    
    def calculate_msd(self, trajectory: np.ndarray, 
                     max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Mean Squared Displacement"""
        n_points = len(trajectory)
        max_lag = n_points // 4 if max_lag is None else min(max_lag, n_points - 1)
        
        # Calculate MSD for different time lags
        msd = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            # Calculate squared displacements
            squared_disp = np.sum((trajectory[lag:] - trajectory[:-lag])**2, axis=1)
            msd[lag-1] = np.mean(squared_disp)
        
        # Convert to physical time
        lag_times = np.arange(1, max_lag + 1) / self.frame_rate
        
        return lag_times, msd
    
    def fit_diffusion_model(self, lag_times: np.ndarray, msd: np.ndarray) -> Dict:
        """Fit diffusion model to MSD data"""
        # Simple linear fit for normal diffusion
        # MSD = 4Dt in 2D
        slope, offset = np.polyfit(lag_times, msd, 1)
        
        # Diffusion coefficient
        D = slope / 4.0
        
        # Check for anomalous diffusion by log-log fit
        log_times = np.log(lag_times)
        log_msd = np.log(msd)
        
        # Linear fit in log-log space
        alpha, log_K = np.polyfit(log_times, log_msd, 1)
        K = np.exp(log_K)
        
        # Return results
        return {
            'diffusion_coefficient': D,
            'anomalous_exponent': alpha,
            'generalized_coefficient': K/4.0,
            'offset': offset
        }
    
    def analyze_tracks(self, tracks: List[Dict], min_length: int = 10) -> pd.DataFrame:
        """Analyze all tracks and compile results"""
        results = []
        
        for track in tracks:
            # Skip tracks that are too short
            if len(track['positions']) < min_length:
                continue
            
            # Convert to physical units
            positions = track['positions'] * self.pixel_size
            frames = track['frames']
            
            # Calculate time in seconds
            times = frames / self.frame_rate
            
            # Calculate MSD
            lag_times, msd = self.calculate_msd(positions)
            
            # Fit diffusion model
            diffusion_params = self.fit_diffusion_model(lag_times, msd)
            
            # Calculate trap stiffness (if applicable)
            var_x = np.var(positions[:, 0])
            var_y = np.var(positions[:, 1])
            
            # For particles in a harmonic trap: k = kT/var
            k_x = self.kT / var_x if var_x > 0 else float('nan')
            k_y = self.kT / var_y if var_y > 0 else float('nan')
            
            # Store results
            result = {
                'track_id': track['id'],
                'length': len(track['positions']),
                'duration': times[-1] - times[0],
                'diffusion_coef': diffusion_params['diffusion_coefficient'],
                'anomalous_exponent': diffusion_params['anomalous_exponent'],
                'trap_stiffness_x': k_x,
                'trap_stiffness_y': k_y,
                'mean_x': np.mean(positions[:, 0]),
                'mean_y': np.mean(positions[:, 1]),
                'var_x': var_x,
                'var_y': var_y
            }
            
            results.append(result)
        
        return pd.DataFrame(results)


class Visualizer:
    """Class for visualizing optical tweezers data and analysis"""
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
    
    def plot_detection(self, image: np.ndarray, centroids: np.ndarray) -> plt.Figure:
        """Plot particle detection results"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Show image
        ax.imshow(image, cmap='gray')
        
        # Plot detected centroids
        if centroids is not None and len(centroids) > 0:
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='r', marker='o', s=80, facecolors='none')
        
        ax.set_title(f"Detected Particles: {len(centroids)}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def plot_tracks(self, image: np.ndarray, 
                   tracks: List[Dict], 
                   tail_length: int = 10) -> plt.Figure:
        """Plot particle tracks"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Show image
        ax.imshow(image, cmap='gray')
        
        # Generate colors for tracks
        colors = plt.cm.jet(np.linspace(0, 1, len(tracks)))
        
        # Plot each track
        for i, track in enumerate(tracks):
            positions = track['positions']
            track_id = track['id']
            
            # Plot full trajectory
            ax.plot(positions[:, 0], positions[:, 1], '-', 
                   color=colors[i], alpha=0.6, linewidth=1)
            
            # Plot end position with track ID
            if len(positions) > 0:
                ax.plot(positions[-1, 0], positions[-1, 1], 'o', 
                       color=colors[i], markersize=8)
                ax.text(positions[-1, 0] + 5, positions[-1, 1] + 5, 
                       str(track_id), color=colors[i])
        
        ax.set_title(f"Particle Tracks: {len(tracks)}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def plot_msd(self, tracks: List[Dict], analyzer: PhysicalAnalyzer) -> plt.Figure:
        """Plot MSD curves for tracks"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate colors for tracks
        colors = plt.cm.jet(np.linspace(0, 1, len(tracks)))
        
        for i, track in enumerate(tracks):
            if len(track['positions']) < 10:
                continue
                
            # Calculate MSD
            positions = track['positions'] * analyzer.pixel_size
            lag_times, msd = analyzer.calculate_msd(positions)
            
            # Fit diffusion model
            params = analyzer.fit_diffusion_model(lag_times, msd)
            
            # Plot MSD data
            ax.plot(lag_times, msd, 'o', color=colors[i], 
                   alpha=0.6, markersize=4, label=f"Track {track['id']}")
            
            # Plot fit
            t_fit = np.linspace(lag_times[0], lag_times[-1], 100)
            msd_fit = 4 * params['diffusion_coefficient'] * t_fit + params['offset']
            ax.plot(t_fit, msd_fit, '-', color=colors[i], linewidth=1)
        
        ax.set_xlabel('Lag Time (s)')
        ax.set_ylabel('MSD (µm²)')
        ax.set_title('Mean Squared Displacement')
        ax.grid(True, alpha=0.3)
        
        if len(tracks) <= 10:  # Only show legend if not too many tracks
            ax.legend(fontsize=8)
        
        return fig
    
    def plot_analysis_summary(self, analysis_df: pd.DataFrame) -> plt.Figure:
        """Plot summary of analysis results"""
        if analysis_df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No analysis data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Diffusion coefficient histogram
        axes[0, 0].hist(analysis_df['diffusion_coef'], bins=10)
        axes[0, 0].set_xlabel('Diffusion Coefficient (µm²/s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Diffusion Coefficient Distribution')
        
        # Anomalous exponent histogram
        axes[0, 1].hist(analysis_df['anomalous_exponent'], bins=10)
        axes[0, 1].set_xlabel('Anomalous Exponent')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Anomalous Exponent Distribution')
        
        # Trap stiffness x vs y
        axes[1, 0].scatter(analysis_df['trap_stiffness_x'], 
                          analysis_df['trap_stiffness_y'])
        axes[1, 0].set_xlabel('Trap Stiffness X (pN/µm)')
        axes[1, 0].set_ylabel('Trap Stiffness Y (pN/µm)')
        axes[1, 0].set_title('Trap Stiffness (X vs Y)')
        
        # Track length histogram
        axes[1, 1].hist(analysis_df['length'], bins=10)
        axes[1, 1].set_xlabel('Track Length (frames)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Track Length Distribution')
        
        plt.tight_layout()
        return fig


def run_pipeline(data_path=None, n_synthetic_frames=100, n_particles=5):
    """Run the complete optical tweezers analysis pipeline"""
    start_time = time.time()
    
    print("Initializing Optical Tweezers Analysis System...")
    
    # Create components
    data_handler = OpticalTweezersData(data_path)
    detector = ParticleDetector(threshold=128, min_size=10)
    tracker = ParticleTracker(max_distance=30)
    analyzer = PhysicalAnalyzer(pixel_size=0.1, frame_rate=30)
    visualizer = Visualizer()
    
    # Generate synthetic data if no data path provided
    if data_path is None:
        print(f"Generating synthetic data with {n_particles} particles over {n_synthetic_frames} frames...")
        images, ground_truth = data_handler.generate_synthetic_data(
            n_frames=n_synthetic_frames,
            n_particles=n_particles
        )
        print(f"Generated {len(images)} frames")
    else:
        print(f"Loading data from {data_path}...")
        images = data_handler.load_image_sequence()
        print(f"Loaded {len(images)} frames")
    
    # Detect particles in each frame
    print("Detecting particles...")
    all_centroids = []
    for i, image in enumerate(images):
        centroids = detector.detect(image)
        all_centroids.append(centroids)
        if i == 0:
            print(f"First frame: detected {len(centroids)} particles")
    
    # Track particles across frames
    print("Tracking particles...")
    tracks = tracker.track(all_centroids)
    print(f"Found {len(tracks)} particle tracks")
    
    # Filter tracks by length
    long_tracks = [t for t in tracks if len(t['positions']) >= 10]
    print(f"Found {len(long_tracks)} tracks of length >= 10 frames")
    
    # Analyze tracks
    print("Analyzing physical properties...")
    analysis_df = analyzer.analyze_tracks(tracks)
    
    # Generate visualizations
    print("Generating visualizations...")
    # Detection visualization
    fig_detection = visualizer.plot_detection(images[0], all_centroids[0])
    
    # Tracking visualization
    fig_tracks = visualizer.plot_tracks(images[0], tracks)
    
    # MSD visualization
    fig_msd = visualizer.plot_msd(long_tracks, analyzer)
    
    # Analysis summary
    fig_summary = visualizer.plot_analysis_summary(analysis_df)
    
    # Save visualizations
    print("Saving results...")
    os.makedirs("results", exist_ok=True)
    fig_detection.savefig("results/detection.png")
    fig_tracks.savefig("results/tracks.png")
    fig_msd.savefig("results/msd.png")
    fig_summary.savefig("results/summary.png")
    
    # Save analysis results
    if not analysis_df.empty:
        analysis_df.to_csv("results/analysis.csv", index=False)
    
    end_time = time.time()
    print(f"Analysis complete in {end_time - start_time:.2f} seconds")
    
    print("\nResults Summary:")
    if not analysis_df.empty:
        print(f"  Number of tracks analyzed: {len(analysis_df)}")
        print(f"  Mean diffusion coefficient: {analysis_df['diffusion_coef'].mean():.4f} µm²/s")
        print(f"  Mean anomalous exponent: {analysis_df['anomalous_exponent'].mean():.4f}")
        if not np.all(np.isnan(analysis_df['trap_stiffness_x'])):
            print(f"  Mean trap stiffness X: {np.nanmean(analysis_df['trap_stiffness_x']):.4e} pN/µm")
            print(f"  Mean trap stiffness Y: {np.nanmean(analysis_df['trap_stiffness_y']):.4e} pN/µm")
    
    print("\nVisualization files saved in the 'results' folder:")
    print("  - detection.png: Particle detection results")
    print("  - tracks.png: Particle tracking results")
    print("  - msd.png: Mean squared displacement analysis")
    print("  - summary.png: Summary of physical properties")
    if not analysis_df.empty:
        print("  - analysis.csv: Complete analysis results")
    
    return {
        'images': images,
        'centroids': all_centroids,
        'tracks': tracks,
        'analysis': analysis_df,
        'figures': {
            'detection': fig_detection,
            'tracks': fig_tracks,
            'msd': fig_msd,
            'summary': fig_summary
        }
    }


if __name__ == "__main__":
    # Run the pipeline with synthetic data
    run_pipeline(n_synthetic_frames=100, n_particles=5)