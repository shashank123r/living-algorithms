# ==========================================
# ast_evolution/evaluator.py
# ==========================================
import numpy as np
from typing import Dict, Tuple, Any
from PIL import Image
import soundfile as sf
from numba import jit
from .genome import Genome

class Evaluator:
    """Handles evaluation and rendering of genomes"""
    
    def __init__(self):
        pass
        
    def create_coordinate_grids(self, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create coordinate grids for evaluation"""
        height, width = size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        return X, Y
        
    def evaluate_genome_image(self, genome: Genome, size: Tuple[int, int] = (256, 256), 
                              t: float = 0.0) -> Dict[str, np.ndarray]:
        """Evaluate genome for image generation"""
        X, Y = self.create_coordinate_grids(size)
        result = {}
        for channel in ['r', 'g', 'b']:
            if channel in genome.trees:
                try:
                    channel_data = genome.trees[channel].evaluate(X, Y, t)
                    channel_data = self._normalize_channel(channel_data)
                    result[channel] = channel_data
                except Exception as e:
                    print(f"Error evaluating {channel} channel: {e}")
                    result[channel] = np.zeros_like(X)
            else:
                result[channel] = np.zeros_like(X)
        return result
        
    def evaluate_genome_audio(self, genome: Genome, duration: float = 2.0, 
                              sample_rate: int = 22050, t_start: float = 0.0) -> Dict[str, np.ndarray]:
        """Evaluate genome for audio generation"""
        num_samples = int(duration * sample_rate)
        t_array = np.linspace(t_start, t_start + duration, num_samples)
        x_coord = np.cos(2 * np.pi * t_array * 0.1)
        y_coord = np.sin(2 * np.pi * t_array * 0.1)
        result = {}
        if 'audio' in genome.trees:
            try:
                audio_data = genome.trees['audio'].evaluate(x_coord, y_coord, t_array)
                audio_data = self._normalize_audio(audio_data)
                result['audio'] = audio_data
            except Exception as e:
                print(f"Error evaluating audio: {e}")
                result['audio'] = np.zeros(num_samples)
        else:
            result['audio'] = np.zeros(num_samples)
        return result
        
    def _normalize_channel(self, data: np.ndarray) -> np.ndarray:
        """Normalize image channel data to [0, 1] range"""
        if np.all(data == data.flat[0]):
            return np.full_like(data, 0.5)
        data_min, data_max = np.min(data), np.max(data)
        if data_max == data_min:
            return np.full_like(data, 0.5)
        normalized = (data - data_min) / (data_max - data_min)
        return np.clip(normalized, 0, 1)
        
    def _normalize_audio(self, data: np.ndarray) -> np.ndarray:
        """Normalize audio data to [-1, 1] range"""
        max_abs = np.max(np.abs(data))
        if max_abs == 0 or not np.isfinite(max_abs):
            return np.zeros_like(data)
        normalized = data / max_abs
        return np.clip(normalized, -1, 1)
        
    def render_image(self, genome: Genome, size: Tuple[int, int] = (512, 512), 
                     t: float = 0.0, filename: str = None) -> Image.Image:
        """Render genome as an image"""
        rgb_data = self.evaluate_genome_image(genome, size, t)
        r_uint8 = (rgb_data['r'] * 255).astype(np.uint8)
        g_uint8 = (rgb_data['g'] * 255).astype(np.uint8)
        b_uint8 = (rgb_data['b'] * 255).astype(np.uint8)
        rgb_array = np.stack([r_uint8, g_uint8, b_uint8], axis=-1)
        img = Image.fromarray(rgb_array, 'RGB')
        if filename:
            img.save(filename)
        return img
        
    def render_audio(self, genome: Genome, duration: float = 2.0, 
                     sample_rate: int = 22050, t_start: float = 0.0, 
                     filename: str = None) -> np.ndarray:
        """Render genome as audio"""
        audio_data = self.evaluate_genome_audio(genome, duration, sample_rate, t_start)
        audio = audio_data['audio']
        if filename:
            sf.write(filename, audio, sample_rate)
        return audio
        
    def create_animation_frames(self, genome: Genome, num_frames: int = 30,
                                size: Tuple[int, int] = (256, 256),
                                t_range: Tuple[float, float] = (0, 2*np.pi)) -> list:
        """Create animation frames by varying t parameter"""
        frames = []
        t_values = np.linspace(t_range[0], t_range[1], num_frames)
        for t in t_values:
            img = self.render_image(genome, size, t)
            frames.append(img)
        return frames
        
    def batch_evaluate_population(self, population, size: Tuple[int, int] = (64, 64),
                                  t: float = 0.0, duration: float = 1.0) -> Dict[int, Dict]:
        """Efficiently evaluate multiple genomes"""
        results = {}
        for i, genome in enumerate(population.genomes):
            try:
                if genome.mode == 'image':
                    evaluation_data = self.evaluate_genome_image(genome, size, t)
                else:
                    evaluation_data = self.evaluate_genome_audio(genome, duration, t_start=t)
                results[i] = evaluation_data
            except Exception as e:
                print(f"Error evaluating genome {i}: {e}")
                if genome.mode == 'image':
                    X, Y = self.create_coordinate_grids(size)
                    results[i] = {'r': np.zeros_like(X), 'g': np.zeros_like(X), 'b': np.zeros_like(X)}
                else:
                    num_samples = int(duration * 22050)
                    results[i] = {'audio': np.zeros(num_samples)}
        return results

# Numba-optimized helper functions for hotspot evaluation
try:
    @jit(nopython=True)
    def safe_divide(a, b):
        """Safe division with protection against division by zero"""
        return np.where(np.abs(b) < 1e-10, np.sign(a), a / b)

    @jit(nopython=True)
    def safe_power(base, exponent):
        """Safe power operation with clipping to avoid overflow"""
        clipped_base = np.clip(base, -100, 100)
        clipped_exp = np.clip(exponent, -10, 10)
        return np.power(np.abs(clipped_base), clipped_exp) * np.sign(clipped_base)

    @jit(nopython=True)
    def safe_sin(x):
        """Numba-optimized sine"""
        return np.sin(x)

    @jit(nopython=True)
    def safe_cos(x):
        """Numba-optimized cosine"""
        return np.cos(x)

except Exception as e:
    print(f"Numba JIT compilation failed: {e}")
