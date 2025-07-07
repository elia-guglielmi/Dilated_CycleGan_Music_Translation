import librosa
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import mean_squared_error
import os
from typing import List, Tuple, Dict
   
import sys
sys.argv = ['']

import os
from frechet_audio_distance import FrechetAudioDistance


class MusicEvaluationMetrics:
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and return audio time series."""
        audio, _ = librosa.load(file_path, sr=self.sr)
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        return mfcc
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features from audio."""
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        return chroma
    
    def compute_mfcc_distance(self, audio1: np.ndarray, audio2: np.ndarray, 
                             metric: str = 'cosine') -> float:
        """
        Compute distance between MFCC features of two audio signals.
        
        Args:
            audio1, audio2: Audio time series
            metric: 'cosine' or 'euclidean'
        
        Returns:
            Distance between MFCC features
        """
        mfcc1 = self.extract_mfcc(audio1)
        mfcc2 = self.extract_mfcc(audio2)
        
        # Take mean across time dimension
        mfcc1_mean = np.mean(mfcc1, axis=1)
        mfcc2_mean = np.mean(mfcc2, axis=1)
        
        if metric == 'cosine':
            return cosine(mfcc1_mean, mfcc2_mean)
        elif metric == 'euclidean':
            return euclidean(mfcc1_mean, mfcc2_mean)
        else:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")
    
    def compute_chroma_distance(self, audio1: np.ndarray, audio2: np.ndarray, 
                               metric: str = 'cosine') -> float:
        """
        Compute distance between chroma features of two audio signals.
        
        Args:
            audio1, audio2: Audio time series
            metric: 'cosine' or 'euclidean'
        
        Returns:
            Distance between chroma features
        """
        chroma1 = self.extract_chroma(audio1)
        chroma2 = self.extract_chroma(audio2)
        
        # Take mean across time dimension
        chroma1_mean = np.mean(chroma1, axis=1)
        chroma2_mean = np.mean(chroma2, axis=1)
        
        if metric == 'cosine':
            return cosine(chroma1_mean, chroma2_mean)
        elif metric == 'euclidean':
            return euclidean(chroma1_mean, chroma2_mean)
        else:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")
    
    def compute_reconstruction_error(self, original_mel: np.ndarray, 
                                   reconstructed_mel: np.ndarray) -> float:
        """
        Compute reconstruction error between original and reconstructed mel spectrograms.
        
        Args:
            original_mel: Original mel spectrogram
            reconstructed_mel: Reconstructed mel spectrogram (after A→B→A)
        
        Returns:
            Mean squared error between spectrograms
        """
        return mean_squared_error(original_mel.flatten(), reconstructed_mel.flatten())
    
    def get_audio_files(self, directory: str, extensions: List[str] = None) -> List[str]:
        """
        Get all audio files from a directory.
        
        Args:
            directory: Path to directory containing audio files
            extensions: List of audio file extensions to include (default: ['.wav', '.mp3', '.flac'])
        
        Returns:
            List of audio file paths
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        
        audio_files = []
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(directory, file))
        
        return sorted(audio_files)
    
    def evaluate_transformation_quality(self, original_dir: str, 
                                      transformed_dir: str, 
                                      target_dir: str) -> Dict[str, float]:
        """
        Evaluate transformation quality using multiple metrics.
        
        Args:
            original_dir: Directory containing original audio files
            transformed_dir: Directory containing transformed audio files
            target_dir: Directory containing target genre audio files
        
        Returns:
            Dictionary with metric results
        """
        # Get audio files from directories
        original_files = self.get_audio_files(original_dir)
        transformed_files = self.get_audio_files(transformed_dir)
        target_files = self.get_audio_files(target_dir)
        
        original_mfcc_distances = []
        original_chroma_distances = []
        mfcc_distances = []
        chroma_distances = []
        
        # Use minimum number of files available
        min_len = min(len(original_files), len(transformed_files), len(target_files))
        
        if min_len == 0:
            raise ValueError("No audio files found in one or more directories")
        
        print(f"Evaluating {min_len} audio files...")
        
        for i in range(min_len):
            # Load audio files
            original_audio=self.load_audio(original_files[i])
            transformed_audio = self.load_audio(transformed_files[i])
            target_audio = self.load_audio(target_files[i])
            
            # Compute MFCC distance (transformed vs target)
            mfcc_dist = self.compute_mfcc_distance(transformed_audio, target_audio)
            mfcc_distances.append(mfcc_dist)
            original_mfcc_dist = self.compute_mfcc_distance(original_audio, target_audio)
            original_mfcc_distances.append(original_mfcc_dist)
            
            # Compute Chroma distance (transformed vs target)
            chroma_dist = self.compute_chroma_distance(transformed_audio, target_audio)
            chroma_distances.append(chroma_dist)
            original_chroma_dist = self.compute_chroma_distance(original_audio, target_audio)
            original_chroma_distances.append(original_chroma_dist)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{min_len} files")
        
        return {
            'mfcc_distance_mean': np.mean(mfcc_distances),
            'mfcc_distance_std': np.std(mfcc_distances),
            'chroma_distance_mean': np.mean(chroma_distances),
            'chroma_distance_std': np.std(chroma_distances),
            'original_mfcc_distance_mean':np.mean(original_mfcc_distances),
            'original_mfcc_distance_std':np.std(original_mfcc_distances),
            'original_chroma_distance_mean': np.mean(original_chroma_distances),
            'original_chroma_distance_std': np.std(original_chroma_distances),
            'num_samples': min_len
        }
    


# Default model configuration
DEFAULT_FAD_MODEL = "vggish"  # or "pann", "clap", "encodec"

def calculate_fad(
    generated_audio_dir: str,
    real_audio_dir: str,
    model_name: str = DEFAULT_FAD_MODEL,
    num_workers: int = 4
) -> float:
    """
    Calculates the Fréchet Audio Distance (FAD) between two sets of audio files.

    Args:
        generated_audio_dir (str): Path to the directory containing the generated audio files (WAV format).
        real_audio_dir (str): Path to the directory containing the real audio files (WAV format).
        model_name (str): Name of the embedding model to use ('vggish', 'pann', 'clap', or 'encodec').
        num_workers (int): Number of parallel threads to use for processing.

    Returns:
        float: The calculated FAD score. A lower score is better.
    """
    print(f"Calculating FAD using the '{model_name}' model...")
    print(f"Directory with REAL audio: {real_audio_dir}")
    print(f"Directory with GENERATED audio: {generated_audio_dir}")

    # Check if directories exist
    if not os.path.exists(real_audio_dir):
        raise ValueError(f"Real audio directory not found at: {real_audio_dir}")
    if not os.path.exists(generated_audio_dir):
        raise ValueError(f"Generated audio directory not found at: {generated_audio_dir}")

    try:
        # Initialize the FAD calculator with the specified model
        if model_name == "vggish":
            frechet = FrechetAudioDistance(
                model_name="vggish",
                sample_rate=16000,
                use_pca=False,
                use_activation=False,
                verbose=False
            )
        elif model_name == "pann":
            frechet = FrechetAudioDistance(
                model_name="pann",
                sample_rate=16000,
                use_pca=False,
                use_activation=False,
                verbose=False
            )
        elif model_name == "clap":
            frechet = FrechetAudioDistance(
                model_name="clap",
                sample_rate=48000,
                submodel_name="630k-audioset",
                verbose=False,
                enable_fusion=False,
            )
        elif model_name == "encodec":
            frechet = FrechetAudioDistance(
                model_name="encodec",
                sample_rate=48000,
                channels=2,
                verbose=False,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Calculate FAD score
        fad_score = frechet.score(
            background_dir=real_audio_dir,
            eval_dir=generated_audio_dir,
            dtype="float32"
        )
        
        print(f"FAD Score: {fad_score}")
        return fad_score
        
    except Exception as e:
        print(f"An error occurred during FAD calculation: {e}")
        return float('inf')