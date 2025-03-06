"""
PyTorch Implementation of AI-Enhanced Optical Tweezers Tracking and Analysis System

This code provides a framework for tracking and analyzing particles in optical tweezers
experiments using deep learning approaches with PyTorch. It includes:
1. Data loading and preprocessing
2. Particle detection with a U-Net architecture
3. Particle tracking across frames
4. Feature extraction and analysis
5. Real-time processing capabilities
6. Visualization tools

Requirements:
- pytorch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import time
import json
import warnings
from scipy.stats import norm
import threading
import queue
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Union

# Suppress warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net architecture for particle detection"""
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Decoder
        self.dec1 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec3 = DoubleConv(128 + 64, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Decoder path
        x = self.dec1(torch.cat([self.up(x4), x3], dim=1))
        x = self.dec2(torch.cat([self.up(x), x2], dim=1))
        x = self.dec3(torch.cat([self.up(x), x1], dim=1))
        
        # Final convolution
        return torch.sigmoid(self.final_conv(x))

class OpticalTweezersDataset(Dataset):
    """Dataset class for optical tweezers data"""
    def __init__(self, data_path: str, img_size: Tuple[int, int] = (512, 512)):
        self.data_path = data_path
        self.img_size = img_size
        self.sequences = self._get_sequences()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True)
        ])

    def _get_sequences(self) -> List[Dict]:
        sequences = []
        for root, _, files in os.walk(self.data_path):
            img_files = [f for f in files if f.endswith(('.tif', '.png', '.jpg'))]
            if img_files:
                sequences.append({
                    'path': root,
                    'files': sorted(img_files),
                    'length': len(img_files)
                })
        return sequences

    def __len__(self) -> int:
        return sum(seq['length'] for seq in self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Find which sequence and index within sequence
        for seq in self.sequences:
            if idx < seq['length']:
                img_path = os.path.join(seq['path'], seq['files'][idx])
                break
            idx -= seq['length']
        
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Apply transforms
        img = self.transform(img)
        
        return img

    def generate_synthetic_data(self, n_frames: int = 100, n_particles: int = 5, 
                              noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data for testing"""
        h, w = self.img_size
        images = torch.zeros((n_frames, 1, h, w))
        positions = torch.zeros((n_frames, n_particles, 2))
        
        # Initialize particle positions
        init_positions = torch.rand(n_particles, 2) * torch.tensor([w, h])
        
        # Parameters
        trap_stiffness = 0.1
        temperature = 0.2
        particle_radius = 5
        diffusion_coeff = 0.5
        
        # Generate frames
        for frame in range(n_frames):
            current_positions = (init_positions.clone() if frame == 0 
                               else current_positions + force + diffusion_coeff * torch.randn_like(init_positions))
            
            # Harmonic trap force
            force = -trap_stiffness * (current_positions - init_positions)
            
            # Ensure particles stay within bounds
            current_positions.clamp_(0, torch.tensor([w-1, h-1]))
            
            # Draw particles
            img = torch.zeros((h, w))
            for x, y in current_positions:
                x_int, y_int = int(x.item()), int(y.item())
                for dy in range(-particle_radius, particle_radius+1):
                    for dx in range(-particle_radius, particle_radius+1):
                        if (y_int+dy >= 0 and y_int+dy < h and 
                            x_int+dx >= 0 and x_int+dx < w and 
                            dx**2 + dy**2 <= particle_radius**2):
                            intensity = torch.exp(torch.tensor(-(dx**2 + dy**2) / 
                                                             (2 * (particle_radius/2)**2)))
                            img[y_int+dy, x_int+dx] = max(img[y_int+dy, x_int+dx].item(), 
                                                        intensity.item())
            
            # Add noise
            img += torch.randn_like(img) * noise_level
            img.clamp_(0, 1)
            
            # Store results
            images[frame, 0] = img
            positions[frame] = current_positions
        
        return images, positions

class ParticleDetector:
    """Class for particle detection using PyTorch model"""
    def __init__(self, model: nn.Module = None):
        self.model = model if model is not None else UNet().to(device)
        self.model.eval()

    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None, 
              epochs: int = 50, 
              lr: float = 1e-4) -> Dict:
        """Train the detection model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        images, masks = batch
                        images, masks = images.to(device), masks.to(device)
                        outputs = self.model(images)
                        val_loss += criterion(outputs, masks).item()
                
                history['val_loss'].append(val_loss / len(val_loader))
            
            history['train_loss'].append(train_loss / len(train_loader))
            
        return history

    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> torch.Tensor:
        """Detect particles in an image"""
        self.model.eval()
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        mask = self.model(image)
        return mask[0, 0].cpu()

    def save(self, filepath: str):
        """Save model"""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        """Load model"""
        self.model.load_state_dict(torch.load(filepath))

class ParticleTracker:
    """Class for tracking particles across frames"""
    def __init__(self, max_distance: float = 30):
        self.max_distance = max_distance
        self.tracks: List[Dict] = []
        
    def detect_particles(self, mask: torch.Tensor, 
                        threshold: float = 0.5, 
                        min_size: int = 10) -> np.ndarray:
        """Detect particles in a segmentation mask"""
        # Convert PyTorch tensor to numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Threshold the mask
        binary = (mask > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter out small components and background
        valid_indices = np.where((stats[:, 4] >= min_size) & (np.arange(num_labels) > 0))[0]
        
        return centroids[valid_indices] if len(valid_indices) > 0 else np.empty((0, 2))
    
    def track_particles(self, centroids_list: List[np.ndarray]) -> List[Dict]:
        """Track particles across multiple frames"""
        self.tracks = []
        active_tracks = []
        
        # Process first frame
        if len(centroids_list) == 0 or len(centroids_list[0]) == 0:
            return self.tracks
        
        for i, pos in enumerate(centroids_list[0]):
            track = {
                'id': i,
                'positions': [pos],
                'frames': [0],
                'last_seen': 0
            }
            active_tracks.append(track)
            self.tracks.append(track)
        
        # Process subsequent frames using Hungarian algorithm
        for frame_idx in range(1, len(centroids_list)):
            current_centroids = centroids_list[frame_idx]
            
            if len(current_centroids) == 0:
                continue
            
            if len(active_tracks) == 0:
                for i, pos in enumerate(current_centroids):
                    track = {
                        'id': len(self.tracks),
                        'positions': [pos],
                        'frames': [frame_idx],
                        'last_seen': frame_idx
                    }
                    active_tracks.append(track)
                    self.tracks.append(track)
                continue
            
            # Calculate cost matrix
            cost_matrix = np.zeros((len(active_tracks), len(current_centroids)))
            for i, track in enumerate(active_tracks):
                last_pos = track['positions'][-1]
                for j, pos in enumerate(current_centroids):
                    cost_matrix[i, j] = np.sqrt(np.sum((last_pos - pos)**2))
            
            # Hungarian algorithm assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            assigned_tracks = set()
            assigned_detections = set()
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= self.max_distance:
                    track = active_tracks[row]
                    track['positions'].append(current_centroids[col])
                    track['frames'].append(frame_idx)
                    track['last_seen'] = frame_idx
                    assigned_tracks.add(row)
                    assigned_detections.add(col)
            
            # Handle unassigned detections
            for j in range(len(current_centroids)):
                if j not in assigned_detections:
                    track = {
                        'id': len(self.tracks),
                        'positions': [current_centroids[j]],
                        'frames': [frame_idx],
                        'last_seen': frame_idx
                    }
                    active_tracks.append(track)
                    self.tracks.append(track)
            
            # Update active tracks
            active_tracks = [track for i, track in enumerate(active_tracks)
                           if i in assigned_tracks or frame_idx - track['last_seen'] <= 5]
        
        # Convert lists to numpy arrays
        for track in self.tracks:
            track['positions'] = np.array(track['positions'])
        
        return self.tracks

class PhysicalAnalyzer:
    """Class for physical analysis of particle trajectories"""
    def __init__(self):
        pass
    
    def calculate_msd(self, trajectory: np.ndarray, 
                     max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Mean Squared Displacement"""
        n_frames = len(trajectory)
        max_lag = n_frames // 4 if max_lag is None else min(max_lag, n_frames - 1)
        
        msd = np.zeros(max_lag)
        counts = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            displacements = trajectory[lag:] - trajectory[:-lag]
            squared_displacements = np.sum(displacements**2, axis=1)
            msd[lag-1] = np.mean(squared_displacements)
            counts[lag-1] = len(squared_displacements)
        
        lag_times = np.arange(1, max_lag + 1)
        return lag_times, msd
    
    def fit_diffusion_model(self, lag_times: np.ndarray, 
                           msd: np.ndarray) -> Dict:
        """Fit diffusion models to MSD data"""
        # Normal diffusion fit
        linear_fit = np.polyfit(lag_times, msd, 1)
        D_normal = linear_fit[0] / 4
        
        # Anomalous diffusion fit
        log_lag = np.log(lag_times)
        log_msd = np.log(msd)
        log_fit = np.polyfit(log_lag, log_msd, 1)
        alpha = log_fit[0]
        D_anomalous = np.exp(log_fit[1]) / 4
        
        # Compare fits
        linear_residuals = msd - (4 * D_normal * lag_times)
        linear_error = np.sum(linear_residuals**2)
        
        anomalous_fit = 4 * D_anomalous * lag_times**alpha
        anomalous_residuals = msd - anomalous_fit
        anomalous_error = np.sum(anomalous_residuals**2)
        
        if anomalous_error < linear_error and 0.7 <= alpha <= 1.3:
            return {
                'D': D_anomalous,
                'alpha': alpha,
                'model': 'anomalous'
            }
        else:
            return {
                'D': D_normal,
                'alpha': 1.0,
                'model': 'normal'
            }
    
    def analyze_trajectories(self, tracks: List[Dict], 
                           pixel_size: float = 0.1, 
                           frame_rate: float = 30) -> pd.DataFrame:
        """Analyze physical properties of all trajectories"""
        results = []
        
        for track in tracks:
            if len(track['positions']) < 10:
                continue
            
            positions = track['positions'] * pixel_size
            lag_times = np.array(track['frames'])
            lag_times = (lag_times - lag_times[0]) / frame_rate
            
            # Calculate MSD and fit diffusion model
            lag_times_msd, msd = self.calculate_msd(positions)
            lag_times_msd = lag_times_msd / frame_rate
            diffusion_params = self.fit_diffusion_model(lag_times_msd, msd)
            
            # Calculate trap stiffness
            var_x = np.var(positions[:, 0])
            var_y = np.var(positions[:, 1])
            kT = 4.11e-21  # Room temperature in Joules
            k_x = kT / var_x if var_x > 0 else None
            k_y = kT / var_y if var_y > 0 else None
            
            results.append({
                'track_id': track['id'],
                'n_frames': len(track['frames']),
                'diffusion_coeff': diffusion_params['D'],
                'alpha': diffusion_params['alpha'],
                'diffusion_model': diffusion_params['model'],
                'trap_stiffness_x': k_x,
                'trap_stiffness_y': k_y,
                'mean_x': np.mean(positions[:, 0]),
                'mean_y': np.mean(positions[:, 1]),
                'var_x': var_x,
                'var_y': var_y
            })
        
        return pd.DataFrame(results)

class RealTimeProcessor:
    """Class for real-time processing of optical tweezers data"""
    def __init__(self, detector: ParticleDetector, 
                 tracker: ParticleTracker, 
                 analyzer: PhysicalAnalyzer, 
                 buffer_size: int = 10):
        self.detector = detector
        self.tracker = tracker
        self.analyzer = analyzer
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.centroid_buffer = []
        self.current_tracks = []
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
    
    def process_frame(self, frame: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Process a single frame"""
        # Convert numpy array to tensor if needed
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 2:
                frame = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
            elif len(frame.shape) == 3:
                frame = torch.from_numpy(frame).float().unsqueeze(0)
        
        # Normalize if needed
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        # Detect particles
        mask = self.detector.detect(frame)
        centroids = self.tracker.detect_particles(mask)
        
        return centroids
    
class Visualizer:
    """Class for visualizing optical tweezers data and analysis"""
    def __init__(self, fig_size: Tuple[int, int] = (12, 10)):
        self.fig_size = fig_size
        
    def plot_detection_results(self, image: torch.Tensor, 
                             mask: torch.Tensor, 
                             centroids: np.ndarray) -> plt.Figure:
        """Plot detection results with original image, mask, and detected particles"""
        # Convert tensors to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        if len(image.shape) == 3:
            image = image[0]  # Remove channel dimension
            
        fig, axes = plt.subplots(1, 3, figsize=self.fig_size)
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot detection mask
        axes[1].imshow(mask, cmap='viridis')
        axes[1].set_title('Detection Mask')
        axes[1].axis('off')
        
        # Plot detected particles
        axes[2].imshow(image, cmap='gray')
        if centroids is not None and len(centroids) > 0:
            axes[2].scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x')
        axes[2].set_title('Detected Particles')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_tracks(self, image: Union[np.ndarray, torch.Tensor], 
                   tracks: List[Dict], 
                   tail_length: int = 10) -> plt.Figure:
        """Plot particle tracks with temporal color coding"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 3:
            image = image[0]
            
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.imshow(image, cmap='gray')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(tracks)))
        for track, color in zip(tracks, colors):
            positions = track['positions']
            
            # Plot full trajectory
            ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=0.5, linewidth=1)
            
            # Plot recent positions with decreasing alpha
            if len(positions) > tail_length:
                recent_positions = positions[-tail_length:]
                alphas = np.linspace(0.2, 1.0, tail_length)
                for pos, alpha in zip(recent_positions, alphas):
                    ax.scatter(pos[0], pos[1], color=color, alpha=alpha, s=20)
            else:
                ax.scatter(positions[:, 0], positions[:, 1], color=color, alpha=0.8, s=20)
        
        ax.set_title('Particle Tracks')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def plot_analysis_results(self, analysis_results: pd.DataFrame) -> plt.Figure:
        """Plot comprehensive analysis results"""
        fig = plt.figure(figsize=self.fig_size)
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Diffusion coefficient distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(analysis_results['diffusion_coeff'], bins='auto')
        ax1.set_xlabel('Diffusion Coefficient')
        ax1.set_ylabel('Count')
        ax1.set_title('Diffusion Coefficient Distribution')
        
        # Anomalous exponent distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(analysis_results['alpha'], bins='auto')
        ax2.set_xlabel('Anomalous Exponent')
        ax2.set_ylabel('Count')
        ax2.set_title('Anomalous Exponent Distribution')
        
        # Trap stiffness correlation
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(analysis_results['trap_stiffness_x'], 
                   analysis_results['trap_stiffness_y'])
        ax3.set_xlabel('Trap Stiffness X')
        ax3.set_ylabel('Trap Stiffness Y')
        ax3.set_title('Trap Stiffness Distribution')
        ax3.grid(True)
        
        # Trajectory lengths
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(analysis_results['n_frames'], bins='auto')
        ax4.set_xlabel('Trajectory Length (frames)')
        ax4.set_ylabel('Count')
        ax4.set_title('Trajectory Length Distribution')
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, frames: List[Union[np.ndarray, torch.Tensor]], 
                        tracks: List[Dict], 
                        interval: int = 50) -> animation.Animation:
        """Create animation of particle tracking"""
        if isinstance(frames[0], torch.Tensor):
            frames = [f.cpu().numpy() if len(f.shape) == 2 
                     else f[0].cpu().numpy() for f in frames]
            
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        def init():
            ax.clear()
            return []
        
        def animate(i):
            ax.clear()
            ax.imshow(frames[i], cmap='gray')
            
            plot_objects = []
            colors = plt.cm.rainbow(np.linspace(0, 1, len(tracks)))
            
            for track, color in zip(tracks, colors):
                frames_array = np.array(track['frames'])
                mask = frames_array <= i
                
                if np.any(mask):
                    positions = track['positions'][mask]
                    if len(positions) > 1:
                        line, = ax.plot(positions[:, 0], positions[:, 1], 
                                      color=color, alpha=0.5)
                        plot_objects.append(line)
                    if len(positions) > 0:
                        scatter = ax.scatter(positions[-1, 0], positions[-1, 1], 
                                          color=color, s=50)
                        plot_objects.append(scatter)
            
            ax.set_title(f'Frame {i}')
            ax.axis('off')
            return plot_objects
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(frames), interval=interval, 
                                     blit=True)
        plt.close()
        return anim

def main():
    """Main execution function"""
    print("Initializing PyTorch Optical Tweezers Analysis System...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize components
    data_handler = OpticalTweezersDataset("./data")
    detector = ParticleDetector()
    tracker = ParticleTracker(max_distance=30)
    analyzer = PhysicalAnalyzer()
    visualizer = Visualizer(fig_size=(15, 5))
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_frames = 100
    n_particles = 5
    images, true_positions = data_handler.generate_synthetic_data(
        n_frames=n_frames,
        n_particles=n_particles,
        noise_level=0.1
    )
    
    # Process frames
    print("Processing frames...")
    all_centroids = []
    all_masks = []
    for frame in images:
        mask = detector.detect(frame)
        centroids = tracker.detect_particles(mask)
        all_centroids.append(centroids)
        all_masks.append(mask)
    
    # Track particles
    print("Tracking particles...")
    tracks = tracker.track_particles(all_centroids)
    
    # Analyze trajectories
    print("Analyzing trajectories...")
    analysis_results = analyzer.analyze_trajectories(
        tracks=tracks,
        pixel_size=0.1,
        frame_rate=30
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Detection results
    fig_detection = visualizer.plot_detection_results(
        image=images[0],
        mask=all_masks[0],
        centroids=all_centroids[0]
    )
    fig_detection.savefig('detection_results.png')
    plt.close(fig_detection)
    
    # Tracking results
    fig_tracks = visualizer.plot_tracks(
        image=images[0],
        tracks=tracks,
        tail_length=10
    )
    fig_tracks.savefig('particle_tracks.png')
    plt.close(fig_tracks)
    
    # Analysis results
    fig_analysis = visualizer.plot_analysis_results(analysis_results)
    fig_analysis.savefig('analysis_results.png')
    plt.close(fig_analysis)
    
    # Animation
    print("Creating animation...")
    anim = visualizer.create_animation(
        frames=images,
        tracks=tracks,
        interval=50
    )
    anim.save('particle_tracking.gif', writer='pillow')
    
    print("\nAnalysis complete! Output files generated:")
    print("- detection_results.png")
    print("- particle_tracks.png")
    print("- analysis_results.png")
    print("- particle_tracking.gif")
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print(f"Number of tracks: {len(tracks)}")
    print(f"Average track length: {analysis_results['n_frames'].mean():.2f} frames")
    print(f"Average diffusion coefficient: {analysis_results['diffusion_coeff'].mean():.2e}")
    print(f"Average anomalous exponent: {analysis_results['alpha'].mean():.2f}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
