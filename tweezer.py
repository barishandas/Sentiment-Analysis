"""
AI-Enhanced Optical Tweezers Tracking and Analysis System

This code provides a framework for tracking and analyzing particles in optical tweezers
experiments using deep learning approaches. It includes:
1. Data loading and preprocessing
2. Particle detection with a U-Net architecture
3. Particle tracking across frames
4. Feature extraction and analysis
5. Real-time processing capabilities
6. Visualization tools

Requirements:
- tensorflow >= 2.6.0
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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

# Suppress warnings
warnings.filterwarnings('ignore')

class OpticalTweezersDataset:
    """Class for handling optical tweezers microscopy datasets"""
    
    def __init__(self, data_path, img_size=(512, 512)):
        """
        Initialize the dataset handler
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing image sequences
        img_size : tuple
            Target size for image resizing (height, width)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.sequences = self._get_sequences()
        
    def _get_sequences(self):
        """Find all image sequences in the data directory"""
        sequences = []
        for root, dirs, files in os.walk(self.data_path):
            img_files = [f for f in files if f.endswith(('.tif', '.png', '.jpg'))]
            if img_files:
                sequences.append({
                    'path': root,
                    'files': sorted(img_files),
                    'length': len(img_files)
                })
        return sequences
    
    def load_sequence(self, seq_idx=0):
        """
        Load a specific image sequence
        
        Parameters:
        -----------
        seq_idx : int
            Index of the sequence to load
            
        Returns:
        --------
        images : ndarray
            4D array of shape (n_frames, height, width, channels)
        """
        if seq_idx >= len(self.sequences):
            raise IndexError(f"Sequence index {seq_idx} out of range")
        
        seq = self.sequences[seq_idx]
        images = []
        
        for img_file in seq['files']:
            img_path = os.path.join(seq['path'], img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Resize image if needed
            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            
            # Normalize pixel values to range [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add channel dimension
            img = np.expand_dims(img, axis=-1)
            images.append(img)
        
        return np.array(images)
    
    def generate_synthetic_data(self, n_frames=100, n_particles=5, noise_level=0.1):
        """
        Generate synthetic optical tweezers data for training or testing
        
        Parameters:
        -----------
        n_frames : int
            Number of frames to generate
        n_particles : int
            Number of particles to simulate
        noise_level : float
            Level of noise to add to images
            
        Returns:
        --------
        images : ndarray
            4D array of shape (n_frames, height, width, channels)
        positions : ndarray
            3D array of shape (n_frames, n_particles, 2) with particle positions
        """
        h, w = self.img_size
        images = np.zeros((n_frames, h, w, 1), dtype=np.float32)
        positions = np.zeros((n_frames, n_particles, 2), dtype=np.float32)
        
        # Initialize particle positions
        init_positions = np.random.rand(n_particles, 2) * np.array([w, h])
        
        # Parameters for particle movement
        trap_stiffness = 0.1
        temperature = 0.2
        particle_radius = 5
        diffusion_coeff = 0.5
        
        # Generate frames
        for frame in range(n_frames):
            # Create blank frame
            img = np.zeros((h, w), dtype=np.float32)
            
            # Update particle positions with Brownian motion in a harmonic trap
            if frame == 0:
                current_positions = init_positions.copy()
            else:
                # Harmonic trap force
                force = -trap_stiffness * (current_positions - init_positions)
                
                # Random displacement from thermal motion
                random_force = np.random.normal(0, np.sqrt(temperature), (n_particles, 2))
                
                # Update positions
                current_positions += force + diffusion_coeff * random_force
                
                # Ensure particles stay within bounds
                current_positions = np.clip(current_positions, 0, [w-1, h-1])
            
            # Draw particles on the image
            for i, (x, y) in enumerate(current_positions):
                # Convert to int for indexing
                x_int, y_int = int(x), int(y)
                
                # Draw a Gaussian spot for each particle
                y_grid, x_grid = np.ogrid[-particle_radius:particle_radius+1, -particle_radius:particle_radius+1]
                mask = x_grid**2 + y_grid**2 <= particle_radius**2
                
                # Apply mask to the image, ensuring we stay within bounds
                for dy in range(-particle_radius, particle_radius+1):
                    for dx in range(-particle_radius, particle_radius+1):
                        if y_int+dy >= 0 and y_int+dy < h and x_int+dx >= 0 and x_int+dx < w:
                            if dx**2 + dy**2 <= particle_radius**2:
                                # Gaussian intensity profile
                                intensity = np.exp(-(dx**2 + dy**2) / (2 * (particle_radius/2)**2))
                                img[y_int+dy, x_int+dx] = max(img[y_int+dy, x_int+dx], intensity)
            
            # Add noise
            img += np.random.normal(0, noise_level, img.shape)
            img = np.clip(img, 0, 1)
            
            # Store the frame and positions
            images[frame, :, :, 0] = img
            positions[frame] = current_positions
        
        return images, positions

class ParticleDetectionModel:
    """U-Net based model for particle detection in microscopy images"""
    
    def __init__(self, input_shape=(512, 512, 1)):
        """
        Initialize the U-Net model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = self._build_unet()
        
    def _build_unet(self):
        """Build a U-Net architecture for particle segmentation"""
        # Input
        inputs = Input(self.input_shape)
        
        # Encoder (downsampling path)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bridge
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # Decoder (upsampling path)
        up5 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
        concat5 = concatenate([up5, conv3], axis=3)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(concat5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
        
        up6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
        concat6 = concatenate([up6, conv2], axis=3)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
        
        up7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
        concat7 = concatenate([up7, conv1], axis=3)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
        
        # Output layer
        outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train(self, train_images, train_masks, validation_split=0.2, batch_size=16, epochs=50):
        """
        Train the particle detection model
        
        Parameters:
        -----------
        train_images : ndarray
            4D array of training images (samples, height, width, channels)
        train_masks : ndarray
            4D array of training masks (samples, height, width, channels)
        validation_split : float
            Fraction of data to use for validation
        batch_size : int
            Batch size for training
        epochs : int
            Number of epochs to train
            
        Returns:
        --------
        history : History object
            Training history
        """
        # Define callbacks
        callbacks = [
            ModelCheckpoint('best_particle_detection_model.h5', save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=10, monitor='val_loss')
        ]
        
        # Train the model
        history = self.model.fit(
            train_images, train_masks,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, images):
        """
        Predict particle masks from input images
        
        Parameters:
        -----------
        images : ndarray
            4D array of input images (samples, height, width, channels)
            
        Returns:
        --------
        masks : ndarray
            4D array of predicted masks (samples, height, width, channels)
        """
        return self.model.predict(images)
    
    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load the model from disk"""
        self.model = tf.keras.models.load_model(filepath)

class ParticleTracker:
    """Class for tracking particles across frames"""
    
    def __init__(self, max_distance=30):
        """
        Initialize the particle tracker
        
        Parameters:
        -----------
        max_distance : float
            Maximum distance between particles in consecutive frames to be considered the same particle
        """
        self.max_distance = max_distance
        self.tracks = []
        
    def detect_particles(self, mask, threshold=0.5, min_size=10):
        """
        Detect particles in a segmentation mask
        
        Parameters:
        -----------
        mask : ndarray
            2D segmentation mask
        threshold : float
            Threshold value for binarizing the mask
        min_size : int
            Minimum particle size in pixels
            
        Returns:
        --------
        centroids : ndarray
            Array of particle centroids (N, 2)
        """
        # Threshold the mask
        binary = (mask > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter out small components and background (label 0)
        valid_indices = np.where((stats[:, 4] >= min_size) & (np.arange(num_labels) > 0))[0]
        
        if len(valid_indices) == 0:
            return np.empty((0, 2))
        
        # Return centroids of valid components
        return centroids[valid_indices]
    
    def track_particles(self, centroids_list):
        """
        Track particles across multiple frames
        
        Parameters:
        -----------
        centroids_list : list of ndarray
            List of centroids arrays for each frame
            
        Returns:
        --------
        tracks : list of dict
            List of particle tracks, each with frame-by-frame positions
        """
        self.tracks = []
        active_tracks = []
        
        # Process first frame
        for i, pos in enumerate(centroids_list[0]):
            track = {
                'id': i,
                'positions': [pos],
                'frames': [0],
                'last_seen': 0
            }
            active_tracks.append(track)
            self.tracks.append(track)
        
        # Process subsequent frames
        for frame_idx in range(1, len(centroids_list)):
            current_centroids = centroids_list[frame_idx]
            
            if len(current_centroids) == 0:
                continue
                
            if len(active_tracks) == 0:
                # If no active tracks, create new ones
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
            
            # Calculate cost matrix (distances between all pairs of points)
            cost_matrix = np.zeros((len(active_tracks), len(current_centroids)))
            
            for i, track in enumerate(active_tracks):
                last_pos = track['positions'][-1]
                for j, pos in enumerate(current_centroids):
                    distance = np.sqrt(np.sum((last_pos - pos)**2))
                    cost_matrix[i, j] = distance
            
            # Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Mark all tracks as unassigned initially
            assigned_tracks = set()
            assigned_detections = set()
            
            # Assign detections to tracks
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= self.max_distance:
                    track = active_tracks[row]
                    track['positions'].append(current_centroids[col])
                    track['frames'].append(frame_idx)
                    track['last_seen'] = frame_idx
                    assigned_tracks.add(row)
                    assigned_detections.add(col)
            
            # Handle unassigned detections (new tracks)
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
            
            # Update active_tracks list (remove tracks that were not assigned)
            active_tracks = [track for i, track in enumerate(active_tracks) 
                           if i in assigned_tracks or frame_idx - track['last_seen'] <= 5]
        
        # Convert positions to numpy arrays
        for track in self.tracks:
            track['positions'] = np.array(track['positions'])
        
        return self.tracks
    
    def get_trajectories(self):
        """
        Get trajectory data for all tracks
        
        Returns:
        --------
        trajectories : pandas.DataFrame
            DataFrame with trajectory data (track_id, frame, x, y)
        """
        if not self.tracks:
            return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y'])
        
        data = []
        for track in self.tracks:
            for frame, pos in zip(track['frames'], track['positions']):
                data.append({
                    'track_id': track['id'],
                    'frame': frame,
                    'x': pos[0],
                    'y': pos[1]
                })
        
        return pd.DataFrame(data)

class PhysicalAnalyzer:
    """Class for physical analysis of particle trajectories"""
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    def calculate_msd(self, trajectory, max_lag=None):
        """
        Calculate Mean Squared Displacement for a single trajectory
        
        Parameters:
        -----------
        trajectory : ndarray
            Array of shape (n_frames, 2) with particle positions
        max_lag : int or None
            Maximum lag time to consider
            
        Returns:
        --------
        lag_times : ndarray
            Array of lag times
        msd : ndarray
            Array of MSD values
        """
        n_frames = len(trajectory)
        
        if max_lag is None:
            max_lag = n_frames // 4
        else:
            max_lag = min(max_lag, n_frames - 1)
        
        msd = np.zeros(max_lag)
        counts = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            displacements = trajectory[lag:] - trajectory[:-lag]
            squared_displacements = np.sum(displacements**2, axis=1)
            msd[lag-1] = np.mean(squared_displacements)
            counts[lag-1] = len(squared_displacements)
        
        lag_times = np.arange(1, max_lag + 1)
        
        return lag_times, msd
    
    def fit_diffusion_model(self, lag_times, msd):
        """
        Fit a diffusion model to MSD data
        
        Parameters:
        -----------
        lag_times : ndarray
            Array of lag times
        msd : ndarray
            Array of MSD values
            
        Returns:
        --------
        params : dict
            Dictionary with fitted parameters:
            - D: diffusion coefficient
            - alpha: anomalous diffusion exponent
            - model: diffusion model type
        """
        # Fit normal diffusion (MSD = 4*D*t)
        linear_fit = np.polyfit(lag_times, msd, 1)
        D_normal = linear_fit[0] / 4
        
        # Fit anomalous diffusion (MSD = 4*D*t^alpha)
        log_lag = np.log(lag_times)
        log_msd = np.log(msd)
        log_fit = np.polyfit(log_lag, log_msd, 1)
        alpha = log_fit[0]
        D_anomalous = np.exp(log_fit[1]) / 4
        
        # Determine which model fits better
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
    
    def extract_trap_stiffness(self, trajectory, temperature=300):
        """
        Extract trap stiffness from a trajectory using equipartition theorem
        
        Parameters:
        -----------
        trajectory : ndarray
            Array of shape (n_frames, 2) with particle positions
        temperature : float
            Temperature in Kelvin
            
        Returns:
        --------
        k_x : float
            Trap stiffness in x direction
        k_y : float
            Trap stiffness in y direction
        """
        # Boltzmann constant in J/K
        kB = 1.380649e-23
        
        # Calculate variances
        var_x = np.var(trajectory[:, 0])
        var_y = np.var(trajectory[:, 1])
        
        # Calculate trap stiffness (k = kB * T / var)
        # Note: This would need proper unit conversions in a real application
        # Here we return in arbitrary units
        k_x = kB * temperature / var_x if var_x > 0 else None
        k_y = kB * temperature / var_y if var_y > 0 else None
        
        return k_x, k_y
    
    def analyze_all_trajectories(self, tracks, pixel_size=0.1, frame_rate=30):
        """
        Analyze physical properties of all trajectories
        
        Parameters:
        -----------
        tracks : list of dict
            List of particle tracks as returned by ParticleTracker
        pixel_size : float
            Size of a pixel in micrometers
        frame_rate : float
            Frame rate in frames per second
            
        Returns:
        --------
        results : pandas.DataFrame
            DataFrame with analysis results for each track
        """
        results = []
        
        for track in tracks:
            # Skip short trajectories
            if len(track['positions']) < 10:
                continue
                
            positions = track['positions'] * pixel_size  # Convert to micrometers
            lag_times = np.array(track['frames'])
            lag_times = (lag_times - lag_times[0]) / frame_rate  # Convert to seconds
            
            # Calculate MSD
            lag_times_msd, msd = self.calculate_msd(positions)
            lag_times_msd = lag_times_msd / frame_rate  # Convert to seconds
            
            # Fit diffusion model
            diffusion_params = self.fit_diffusion_model(lag_times_msd, msd)
            
            # Extract trap stiffness
            k_x, k_y = self.extract_trap_stiffness(positions)
            
            # Store results
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
                'var_x': np.var(positions[:, 0]),
                'var_y': np.var(positions[:, 1])
            })
        
        return pd.DataFrame(results)

class RealTimeProcessor:
    """Class for real-time processing of optical tweezers data"""
    
    def __init__(self, detection_model, tracker, analyzer, buffer_size=10):
        """
        Initialize the real-time processor
        
        Parameters:
        -----------
        detection_model : ParticleDetectionModel
            Model for particle detection
        tracker : ParticleTracker
            Tracker for particle trajectories
        analyzer : PhysicalAnalyzer
            Analyzer for physical properties
        buffer_size : int
            Size of the frame buffer
        """
        self.detection_model = detection_model
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
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Parameters:
        -----------
        frame : ndarray
            2D or 3D array with the frame data
            
        Returns:
        --------
        centroids : ndarray
            Array of detected particle centroids
        """
        # Ensure frame has correct shape
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)
        if len(frame.shape) == 3 and frame.shape[2] > 1:
            frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=-1)
        
        # Normalize if needed
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
        
        # Predict mask
        frame_batch = np.expand_dims(frame, axis=0)
        mask = self.detection_model.predict(frame_batch)[0, :, :, 0]
        
        # Detect particles
        centroids = self.tracker.detect_particles(mask)
        
        return centroids
    
    def update_buffer(self, frame, centroids):
        """
        Update the frame and centroid buffers
        
        Parameters:
        -----------
        frame : ndarray
            Current frame
        centroids : ndarray
            Detected centroids
        """
        # Add to buffer
        self.frame_buffer.append(frame)
        self.centroid_buffer.append(centroids)
        
        # Keep buffer size limited
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            self.centroid_buffer.pop(0)
    
    def start_processing(self):
        """Start the processing thread"""
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the processing thread"""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Main processing loop (runs in a separate thread)"""
        while not self.stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame = self.processing_queue.get(timeout=0.1)
                
                # Process frame
                centroids = self.process_frame(frame)
                
                # Update buffer
                self.update_buffer(frame, centroids)
                
                # Update tracks if we have enough frames
                if len(self.centroid_buffer) >= 2:
                    self.current_tracks = self.tracker.track_particles(self.centroid_buffer)
                
                # Analyze tracks and put results in results queue
                if self.current_tracks:
                    results = self.analyzer.analyze_all_trajectories(self.current_tracks)
                    self.results_queue.put({
                        'frame': frame,
                        'centroids': centroids,
                        'tracks': self.current_tracks,
                        'analysis': results
                    })
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                # No frame available, just continue
                continue
    
    def add_frame(self, frame):
        """
        Add a frame to the processing queue
        
        Parameters:
        -----------
        frame : ndarray
            Frame to process
        """
        self.processing_queue.put(frame)
    
    def get_latest_results(self, block=False, timeout=None):
        """
        Get the latest processing results
        
        Parameters:
        -----------
        block : bool
            Whether to block until results are available
        timeout : float or None
            Timeout for blocking
            
        Returns:
        --------
        results : dict or None
            Latest processing results, or None if no results available
        """
        try:
            return self.results_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

class OpticalTweezersVisualizer:
    """Class for visualizing optical tweezers data and analysis"""
    
    def __init__(self, fig_size=(12, 10)):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        fig_size : tuple
            Figure size in inches
        """
        self.fig_size = fig_size
    
    def plot_detection_results(self, image, mask, centroids):
        """
        Plot detection results
        
        Parameters:
        -----------
        image : ndarray
            Original image
        mask : ndarray
            Detection mask
        centroids : ndarray
            Detected particle centroids
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=self.fig_size)
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot detection mask
        axes[1].imshow(mask, cmap='viridis')
        axes[1].set_title('Detection Mask')
        axes[1].axis('off')
        
        # Plot centroids on original image
        axes[2].imshow(image, cmap='gray')
        axes[2].set_title('Detected Particles')
        axes[2].axis('off')
        
        # Plot centroids
