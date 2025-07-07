import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from scipy.ndimage import zoom
import warnings
from IPython.display import Audio, display
warnings.filterwarnings('ignore')

class MusicReconstructor:
    """
    Reconstructs audio from a processed mel spectrogram.
    This class is designed to reverse the steps of the MusicPreprocessor.
    """
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 segment_duration: float = 5.0,
                 global_mean: float = 0.0,
                 global_std: float = 1.0):
        """
        Initialize the reconstructor with preprocessing parameters.
        
        Args:
            sample_rate (int): Sample rate of the audio
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            n_mels (int): Number of mel frequency bins
            segment_duration (float): Duration of audio segments in seconds
            global_mean (float): Global mean for normalization
            global_std (float): Global standard deviation for normalization
        """
        self.params = {
            'sample_rate': sample_rate,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'segment_duration': segment_duration,
            'global_mean': global_mean,
            'global_std': global_std
        }
        
        # Calculate original spectrogram time dimension before resizing
        # This is crucial for reversing the resize step
        segment_samples = int(self.params['segment_duration'] * self.params['sample_rate'])
        self.original_time_steps = int(np.ceil(segment_samples / self.params['hop_length']))

    def _denormalize_spectrogram(self, norm_spec: np.ndarray) -> np.ndarray:
        """Reverses z-score normalization."""
        mean = self.params['global_mean']
        std = self.params['global_std']
        return (norm_spec * std) + mean

    def _unresize_spectrogram(self, resized_spec: np.ndarray) -> np.ndarray:
        """
        Reverses the resizing operation.
        """
        original_n_mels = self.params['n_mels']
        target_shape = (original_n_mels, self.original_time_steps)
        
        # Calculate zoom factors to scale it back up
        zoom_factors = (
            target_shape[0] / resized_spec.shape[0],
            target_shape[1] / resized_spec.shape[1]
        )
        
        # Use scipy.ndimage.zoom, same as in the preprocessor
        unresized_spec = zoom(resized_spec, zoom_factors, order=1)
        return unresized_spec

    def _inverse_log_scale(self, log_spec: np.ndarray) -> np.ndarray:
        """Reverses the log1p operation."""
        return np.expm1(log_spec)

    def _melspectrogram_to_audio(self, mel_spec: np.ndarray) -> np.ndarray:
        """Converts a mel spectrogram back to an audio waveform."""
        # Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.params['sample_rate'],
            n_fft=self.params['n_fft'],
            hop_length=self.params['hop_length'],
            n_iter=60 
        )
        return audio

    def reconstruct_and_save(self, 
                             input_spectrogram: np.ndarray, 
                             output_path: str):
        """
        Performs the full reconstruction pipeline for a single spectrogram
        and saves it as a WAV file.
        """
        print("Step 1: Denormalizing spectrogram...")
        denorm_spec = self._denormalize_spectrogram(input_spectrogram)
        
        print("Step 2: Un-resizing spectrogram...")
        unresized_spec = self._unresize_spectrogram(denorm_spec)
        
        print("Step 3: Reversing log scaling...")
        linear_spec = self._inverse_log_scale(unresized_spec)
        
        # Ensure no negative values after expm1 due to floating point inaccuracies
        linear_spec[linear_spec < 0] = 0
        
        print("Step 4: Inverting mel spectrogram to audio (Griffin-Lim)...")
        audio_waveform = self._melspectrogram_to_audio(linear_spec)
        
        print(f"Step 5: Saving audio to {output_path}...")
        sf.write(output_path, audio_waveform, self.params['sample_rate'])
        print("Reconstruction complete!")

    def reconstruct_and_show(self, 
                            input_spectrogram: np.ndarray):
        """
        Performs the full reconstruction pipeline for a single spectrogram
        and displays it as an audio widget.
        """
        print("Step 1: Denormalizing spectrogram...")
        denorm_spec = self._denormalize_spectrogram(input_spectrogram)
        
        print("Step 2: Un-resizing spectrogram...")
        unresized_spec = self._unresize_spectrogram(denorm_spec)
        
        print("Step 3: Reversing log scaling...")
        linear_spec = self._inverse_log_scale(unresized_spec)
        
        # Ensure no negative values after expm1 due to floating point inaccuracies
        linear_spec[linear_spec < 0] = 0
        
        print("Step 4: Inverting mel spectrogram to audio (Griffin-Lim)...")
        audio_waveform = self._melspectrogram_to_audio(linear_spec)
        
        display(Audio(audio_waveform, rate=self.params['sample_rate']))