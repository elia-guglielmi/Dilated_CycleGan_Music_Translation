import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import h5py
from typing import Tuple, List, Optional
import warnings

try:
    from scipy.ndimage import zoom
except ImportError:
    print("Warning: scipy is not installed. The 'resize_spectrogram' method will not work.")
    zoom = None

warnings.filterwarnings('ignore')

class MusicPreprocessor:
    """
    Advanced preprocessing pipeline for CycleGAN music genre transformation.
    Generates 2-channel spectrograms (Mel Spectrogram + Constant-Q Transform).
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 segment_duration: float = 5.0,
                 overlap_ratio: float = 0.5,
                 # Mel Spectrogram parameters
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 fmin: float = 0.0,
                 fmax: float = 8000.0,
                 # CQT parameters
                 n_cqt_bins: int = 84, # 7 octaves * 12 bins/octave
                 bins_per_octave: int = 12,
                 # General parameters
                 normalize_audio: bool = True,
                 log_scale: bool = True,
                 spectrogram_size: Tuple[int, int] = (128, 512)):
        
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.overlap_ratio = overlap_ratio
        self.segment_samples = int(segment_duration * sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap_ratio))
        
        # Mel spectrogram parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Shared frequency parameters
        self.fmin = fmin
        self.fmax = fmax

        # CQT parameters
        self.n_cqt_bins = n_cqt_bins
        self.bins_per_octave = bins_per_octave
        
        self.normalize_audio = normalize_audio
        self.log_scale = log_scale
        self.spectrogram_size = spectrogram_size
        
        #Statistics for per-channel normalization
        self.global_mel_mean = None
        self.global_mel_std = None
        self.global_cqt_mean = None
        self.global_cqt_std = None
        
    def load_and_preprocess_audio(self, file_path: str) -> Optional[np.ndarray]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            audio = self._trim_silence(audio)
            if self.normalize_audio:
                audio = self._normalize_audio(audio)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _trim_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def _normalize_audio(self, audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        if method == 'peak':
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
        elif method == 'rms':
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms
                peak = np.max(np.abs(audio))
                if peak > 1.0:
                    audio = audio / peak
        return audio
    
    def segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        segments = []
        if len(audio) < self.segment_samples:
            padding = self.segment_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        for start in range(0, len(audio) - self.segment_samples + 1, self.hop_samples):
            segment = audio[start:start + self.segment_samples]
            segments.append(segment)
        return segments
    
    def audio_to_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax, power=2.0
        )
        if self.log_scale:
            mel_spec = np.log1p(mel_spec)
        return mel_spec

    # Method to compute CQT
    def audio_to_cqt(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio segment to Constant-Q Transform spectrogram.
        """
        # Compute CQT. We use hop_length to keep the time dimension consistent
        # with the Mel spectrogram before resizing.
        cqt = librosa.cqt(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length,
            fmin=self.fmin if self.fmin > 0 else librosa.note_to_hz('C1'), # CQT requires a positive fmin
            n_bins=self.n_cqt_bins, bins_per_octave=self.bins_per_octave
        )
        
        # CQT returns complex numbers, so we take the absolute value for magnitude
        cqt_mag = np.abs(cqt)

        if self.log_scale:
            cqt_mag = np.log1p(cqt_mag)
        
        return cqt_mag
    
    def resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        if zoom is None:
            raise ImportError("scipy is not installed. Cannot resize spectrogram.")
        
        current_shape = spectrogram.shape
        # Handle case where spectrogram is empty
        if current_shape[0] == 0 or current_shape[1] == 0:
            return np.zeros(self.spectrogram_size)

        zoom_factors = (
            self.spectrogram_size[0] / current_shape[0],
            self.spectrogram_size[1] / current_shape[1]
        )
        resized = zoom(spectrogram, zoom_factors, order=1) # Using bilinear interpolation
        return resized
    
    # Compute statistics for each channel separately
    def compute_dataset_statistics(self, spectrograms: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute global mean and std for each channel (Mel and CQT).
        `spectrograms` is expected to be a numpy array of shape (N, H, W, 2).
        """
        # Channel 0: Mel Spectrograms
        mel_channel = spectrograms[:, :, :, 0]
        self.global_mel_mean = np.mean(mel_channel)
        self.global_mel_std = np.std(mel_channel)

        # Channel 1: CQTs
        cqt_channel = spectrograms[:, :, :, 1]
        self.global_cqt_mean = np.mean(cqt_channel)
        self.global_cqt_std = np.std(cqt_channel)
        
        print(f"Mel Stats: Mean={self.global_mel_mean:.4f}, Std={self.global_mel_std:.4f}")
        print(f"CQT Stats: Mean={self.global_cqt_mean:.4f}, Std={self.global_cqt_std:.4f}")
        
        return ((self.global_mel_mean, self.global_mel_std), 
                (self.global_cqt_mean, self.global_cqt_std))
    
    # Normalize each channel with its own statistics
    def normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize a 2-channel spectrogram using global statistics.
        `spectrogram` is expected to be of shape (H, W, 2).
        """
        if self.global_mel_mean is None or self.global_cqt_mean is None:
            raise ValueError("Global statistics not computed. Call compute_dataset_statistics first.")
        
        # Create a copy to avoid modifying the original array
        normalized = np.copy(spectrogram)
        
        # Normalize Mel channel
        normalized[:, :, 0] = (normalized[:, :, 0] - self.global_mel_mean) / (self.global_mel_std + 1e-8)
        # Normalize CQT channel
        normalized[:, :, 1] = (normalized[:, :, 1] - self.global_cqt_mean) / (self.global_cqt_std + 1e-8)
        
        return normalized
    
    #Main processing function to generate 2-channel data
    def process_genre_directory(self, genre_dir: str, genre_name: str) -> Tuple[np.ndarray, List[str]]:
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(Path(genre_dir).glob(ext))
        
        all_stacked_spectrograms = []
        file_info = []
        
        print(f"Processing {len(audio_files)} files for genre: {genre_name}")
        
        for i, file_path in enumerate(audio_files):
            print(f"  - Processing {i+1}/{len(audio_files)}: {file_path.name}", end='\r')
            
            audio = self.load_and_preprocess_audio(str(file_path))
            if audio is None:
                continue
            
            segments = self.segment_audio(audio)
            
            for j, segment in enumerate(segments):
                # Compute both representations
                mel_spec = self.audio_to_melspectrogram(segment)
                cqt_spec = self.audio_to_cqt(segment)
                
                # Resize both to the same target size
                mel_spec_resized = self.resize_spectrogram(mel_spec)
                cqt_spec_resized = self.resize_spectrogram(cqt_spec)
                
                # Stack them into a single 2-channel array
                stacked_spec = np.stack([mel_spec_resized, cqt_spec_resized], axis=-1)
                
                all_stacked_spectrograms.append(stacked_spec)
                file_info.append(f"{file_path.stem}_segment_{j}")
        
        print(f"\nFinished processing for {genre_name}. Generated {len(all_stacked_spectrograms)} samples.")
        return np.array(all_stacked_spectrograms), file_info
    
    #Save new CQT parameters
    def save_processed_data(self, spectrograms: np.ndarray, file_info: List[str], 
                          output_path: str, genre_name: str):
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(f'{genre_name}_spectrograms', data=spectrograms, 
                           compression='gzip', compression_opts=9)
            f.create_dataset(f'{genre_name}_file_info', 
                           data=[s.encode('utf-8') for s in file_info])
            
            params = f.create_group(f'{genre_name}_params')
            params.attrs['sample_rate'] = self.sample_rate
            params.attrs['n_fft'] = self.n_fft
            params.attrs['hop_length'] = self.hop_length
            params.attrs['n_mels'] = self.n_mels
            params.attrs['n_cqt_bins'] = self.n_cqt_bins 
            params.attrs['bins_per_octave'] = self.bins_per_octave 
            params.attrs['segment_duration'] = self.segment_duration
            params.attrs['spectrogram_size'] = self.spectrogram_size
            
            if self.global_mel_mean is not None:
                params.attrs['global_mel_mean'] = self.global_mel_mean
                params.attrs['global_mel_std'] = self.global_mel_std
                params.attrs['global_cqt_mean'] = self.global_cqt_mean
                params.attrs['global_cqt_std'] = self.global_cqt_std
    
    # Load per-channel normalization stats
    def load_processed_data(self, file_path: str, genre_name: str) -> Tuple[np.ndarray, List[str]]:
        with h5py.File(file_path, 'r') as f:
            spectrograms = f[f'{genre_name}_spectrograms'][:]
            file_info = [s.decode('utf-8') for s in f[f'{genre_name}_file_info'][:]]
            
            if f'{genre_name}_params' in f:
                params = f[f'{genre_name}_params']
                if 'global_mel_mean' in params.attrs:
                    self.global_mel_mean = params.attrs['global_mel_mean']
                    self.global_mel_std = params.attrs['global_mel_std']
                    self.global_cqt_mean = params.attrs['global_cqt_mean']
                    self.global_cqt_std = params.attrs['global_cqt_std']
        
        return spectrograms, file_info