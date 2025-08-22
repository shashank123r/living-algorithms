"""
ast_evolution/fitness.py - Novelty search and aesthetic fitness evaluation
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cdist
from scipy import ndimage
import librosa

class FeatureExtractor:
    """Extract features from generated content for novelty search"""
    
    @staticmethod
    def extract_image_features(rgb_data: Dict[str, np.ndarray], 
                              size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Extract features from RGB image data"""
        features = []
        
        # Combine RGB channels
        r, g, b = rgb_data['r'], rgb_data['g'], rgb_data['b']
        
        # Normalize to 0-1 range
        r = (r - r.min()) / (r.max() - r.min() + 1e-10)
        g = (g - g.min()) / (g.max() - g.min() + 1e-10)
        b = (b - b.min()) / (b.max() - b.min() + 1e-10)
        
        # Reshape if needed
        if r.shape != size:
            r = np.resize(r, size)
            g = np.resize(g, size)
            b = np.resize(b, size)
            
        # Color statistics
        features.extend([
            np.mean(r), np.std(r), np.min(r), np.max(r),
            np.mean(g), np.std(g), np.min(g), np.max(g),
            np.mean(b), np.std(b), np.min(b), np.max(b)
        ])
        
        # Color correlations
        features.extend([
            np.corrcoef(r.flat, g.flat)[0, 1],
            np.corrcoef(r.flat, b.flat)[0, 1],
            np.corrcoef(g.flat, b.flat)[0, 1]
        ])
        
        # Spatial features - gradients
        for channel in [r, g, b]:
            grad_x = np.gradient(channel, axis=1)
            grad_y = np.gradient(channel, axis=0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(grad_mag),
                np.std(grad_mag),
                np.percentile(grad_mag, 90)
            ])
            
        # Edge density using Sobel filter
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.sum(edge_magnitude > np.percentile(edge_magnitude, 80)) / edge_magnitude.size
        ])
        
        # Frequency domain features (2D FFT)
        fft_gray = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft_gray)
        
        # Radial frequency bins
        center = (size[0] // 2, size[1] // 2)
        y, x = np.ogrid[:size[0], :size[1]]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        for r_bin in [5, 10, 20]:
            mask = (radius <= r_bin) & (radius > r_bin - 5)
            if np.any(mask):
                features.append(np.mean(fft_magnitude[mask]))
            else:
                features.append(0)
                
        # Texture features - Local Binary Pattern approximation
        for channel in [r, g, b]:
            # Simple texture measure: variance of local neighborhoods
            local_var = ndimage.generic_filter(channel, np.var, size=3)
            features.append(np.mean(local_var))
            
        return np.array(features)
        
    @staticmethod
    def extract_audio_features(audio_data: np.ndarray, 
                              sr: int = 22050, 
                              duration: float = 2.0) -> np.ndarray:
        """Extract features from audio data"""
        features = []
        
        # Ensure audio is 1D and clip duration
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
            
        max_samples = int(sr * duration)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        elif len(audio_data) < max_samples:
            audio_data = np.pad(audio_data, (0, max_samples - len(audio_data)))
            
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # Time domain features
        features.extend([
            np.mean(audio_data),
            np.std(audio_data),
            np.sqrt(np.mean(audio_data**2)),  # RMS
            np.max(np.abs(audio_data)),  # Peak amplitude
        ])
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
        features.append(zcr)
        
        # Spectral features
        stft = librosa.stft(audio_data, n_fft=1024, hop_length=512)
        magnitude = np.abs(stft)
        
        # Spectral centroid, bandwidth, rolloff
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)[0]
        
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(13):
            features.extend([
                np.mean(mfccs[i]),
                np.std(mfccs[i])
            ])
            
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(S=magnitude)
        for i in range(min(7, contrast.shape[0])):
            features.append(np.mean(contrast[i]))
            
        # Chroma features
        chroma = librosa.feature.chroma_stft(S=magnitude)
        for i in range(12):
            features.append(np.mean(chroma[i]))
            
        return np.array(features)

class NoveltySearch:
    """Novelty search implementation with feature archive"""
    
    def __init__(self, archive_size: int = 1000, k_neighbors: int = 15):
        self.archive = []  # List of feature vectors
        self.archive_size = archive_size
        self.k_neighbors = k_neighbors
        self.feature_extractor = FeatureExtractor()
        
    def calculate_novelty(self, features: np.ndarray) -> float:
        """Calculate novelty score based on distance to k nearest neighbors"""
        if len(self.archive) < self.k_neighbors:
            # Not enough samples in archive, high novelty
            return 1.0
            
        # Convert archive to array
        archive_array = np.array(self.archive)
        
        # Calculate distances to all archived features
        distances = cdist([features], archive_array, metric='euclidean')[0]
        
        # Get k nearest neighbors
        k_nearest_distances = np.partition(distances, self.k_neighbors)[:self.k_neighbors]
        
        # Novelty is mean distance to k nearest neighbors
        novelty = np.mean(k_nearest_distances)
        
        return novelty
        
    def add_to_archive(self, features: np.ndarray, novelty_threshold: float = 0.1) -> bool:
        """Add features to archive if novel enough"""
        if len(self.archive) == 0:
            self.archive.append(features.copy())
            return True
            
        novelty = self.calculate_novelty(features)
        
        # Add if novel enough or archive not full
        if novelty > novelty_threshold or len(self.archive) < self.archive_size // 2:
            self.archive.append(features.copy())
            
            # Maintain archive size
            if len(self.archive) > self.archive_size:
                # Remove oldest entry
                self.archive.pop(0)
                
            return True
            
        return False

class AestheticEvaluator:
    """Simple aesthetic fitness evaluation"""
    
    @staticmethod
    def evaluate_image_aesthetic(rgb_data: Dict[str, np.ndarray]) -> float:
        """Evaluate aesthetic quality of image"""
        r, g, b = rgb_data['r'], rgb_data['g'], rgb_data['b']
        
        # Normalize channels
        r = (r - r.min()) / (r.max() - r.min() + 1e-10)
        g = (g - g.min()) / (g.max() - g.min() + 1e-10)
        b = (b - b.min()) / (b.max() - b.min() + 1e-10)
        
        aesthetic_score = 0.0
        
        # Color variance (prefer some variation but not chaos)
        color_vars = [np.var(r), np.var(g), np.var(b)]
        target_var = 0.1  # Target variance
        var_score = 1.0 - np.mean([abs(v - target_var) for v in color_vars]) / target_var
        aesthetic_score += 0.3 * max(0, var_score)
        
        # Edge density (prefer some but not too many edges)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = np.mean(edge_magnitude)
        
        # Prefer moderate edge density
        target_edge = 0.3
        edge_score = 1.0 - abs(edge_density - target_edge) / target_edge
        aesthetic_score += 0.4 * max(0, edge_score)
        
        # Color harmony (prefer balanced RGB)
        mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
        balance = 1.0 - np.std([mean_r, mean_g, mean_b])
        aesthetic_score += 0.2 * max(0, balance)
        
        # Smoothness vs detail balance
        smooth_score = 1.0 - np.std(gray)  # Penalize too much variation
        detail_score = np.std(gray)  # Reward some variation
        balance_score = 1.0 - abs(smooth_score - detail_score)
        aesthetic_score += 0.1 * max(0, balance_score)
        
        return np.clip(aesthetic_score, 0, 1)
        
    @staticmethod
    def evaluate_audio_aesthetic(audio_data: np.ndarray, sr: int = 22050) -> float:
        """Evaluate aesthetic quality of audio"""
        if len(audio_data) == 0:
            return 0.0
            
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        aesthetic_score = 0.0
        
        # Dynamic range (prefer some variation)
        dynamic_range = np.max(audio_data) - np.min(audio_data)
        aesthetic_score += 0.2 * min(1.0, dynamic_range)
        
        # Spectral spread (prefer rich harmonic content)
        stft = librosa.stft(audio_data, n_fft=1024, hop_length=512)
        magnitude = np.abs(stft)
        
        if magnitude.size > 0:
            spectral_spread = np.mean(librosa.feature.spectral_bandwidth(S=magnitude))
            # Normalize to 0-1 range (assuming max spread of sr/4)
            spread_score = min(1.0, spectral_spread / (sr / 4))
            aesthetic_score += 0.3 * spread_score
            
            # Spectral centroid stability (prefer some stability)
            centroids = librosa.feature.spectral_centroid(S=magnitude)[0]
            centroid_stability = 1.0 - (np.std(centroids) / (np.mean(centroids) + 1e-10))
            aesthetic_score += 0.2 * max(0, min(1, centroid_stability))
            
        # Rhythmic regularity (basic measure)
        # Simple autocorrelation-based measure
        if len(audio_data) > 1000:
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            if len(autocorr) > 100:
                peak_strength = np.max(autocorr[50:min(500, len(autocorr))])
                rhythm_score = min(1.0, peak_strength / np.max(autocorr[:50]))
                aesthetic_score += 0.15 * rhythm_score
                
        # Harmonic content (prefer some harmonicity)
        try:
            harmonics = librosa.effects.harmonic(audio_data)
            harmonic_ratio = np.sqrt(np.mean(harmonics**2)) / (np.sqrt(np.mean(audio_data**2)) + 1e-10)
            aesthetic_score += 0.15 * min(1.0, harmonic_ratio)
        except:
            pass
            
        return np.clip(aesthetic_score, 0, 1)

class FitnessEvaluator:
    """Combined fitness evaluation using novelty search + aesthetic proxy"""
    
    def __init__(self, novelty_weight: float = 0.7, aesthetic_weight: float = 0.3):
        self.novelty_search = NoveltySearch()
        self.aesthetic_evaluator = AestheticEvaluator()
        self.novelty_weight = novelty_weight
        self.aesthetic_weight = aesthetic_weight
        
    def evaluate_genome(self, genome, evaluation_data: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """Evaluate a genome's fitness, returning (total_fitness, novelty, aesthetic)"""
        
        # Extract features based on mode
        if genome.mode == 'image':
            features = self.novelty_search.feature_extractor.extract_image_features(evaluation_data)
            aesthetic_score = self.aesthetic_evaluator.evaluate_image_aesthetic(evaluation_data)
        else:  # audio
            features = self.novelty_search.feature_extractor.extract_audio_features(evaluation_data['audio'])
            aesthetic_score = self.aesthetic_evaluator.evaluate_audio_aesthetic(evaluation_data['audio'])
            
        # Calculate novelty
        novelty_score = self.novelty_search.calculate_novelty(features)
        
        # Add to archive if novel enough
        self.novelty_search.add_to_archive(features)
        
        # Combined fitness
        total_fitness = (self.novelty_weight * novelty_score + 
                        self.aesthetic_weight * aesthetic_score)
        
        return total_fitness, novelty_score, aesthetic_score
        
    def get_archive_size(self) -> int:
        """Get current archive size"""
        return len(self.novelty_search.archive)
        
    def save_archive(self, filename: str) -> None:
        """Save feature archive to file"""
        np.save(filename, np.array(self.novelty_search.archive))
        
    def load_archive(self, filename: str) -> None:
        """Load feature archive from file"""
        try:
            archive_data = np.load(filename)
            self.novelty_search.archive = [arr for arr in archive_data]
        except Exception as e:
            print(f"Could not load archive: {e}")