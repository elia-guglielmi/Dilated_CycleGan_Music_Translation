import tensorflow as tf
import numpy as np
import h5py
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class MusicDataLoader:
    """
    TensorFlow data loader class.
    Handles 2-channel data (Mel Spectrogram + CQT).
    """
    
    def __init__(self, data_file_a: str, data_file_b: str, batch_size: int = 16, 
                 train_split: float = 0.9, val_split: float = 0.1,
                 buffer_size: int = 1000, prefetch_size: int=tf.data.AUTOTUNE,
                 seed: int = 42):
        
        #Load from separate files for clarity and flexibility
        self.data_file_a = data_file_a
        self.data_file_b = data_file_b
        
        self.batch_size = batch_size
        self.train_split = train_split
        
        self.val_split = val_split 
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size
        self.seed = seed
        
        # Set seeds for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Load data
        self.genre_a_data, self.params_a = self._load_data(self.data_file_a, 'genre_a')
        self.genre_b_data, self.params_b = self._load_data(self.data_file_b, 'genre_b')
        
        # Split data
        self.splits = self._create_splits()
    
    def _load_data(self, data_file: str, genre_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Load processed spectrograms and params from a single HDF5 file.
        """
        print(f"Loading data for {genre_name} from {data_file}...")
        
        with h5py.File(data_file, 'r') as f:
            spectrograms = f[f'{genre_name}_spectrograms'][:]
            
            # The loaded data should already be (N, H, W, 2)
            print(f"Loaded {genre_name}: {spectrograms.shape}")
            if len(spectrograms.shape) != 4 or spectrograms.shape[3] != 2:
                raise ValueError(f"Expected 4D data with 2 channels for {genre_name}, but got shape {spectrograms.shape}")

            params = {}
            if f'{genre_name}_params' in f:
                param_group = f[f'{genre_name}_params']
                for key in param_group.attrs:
                    params[key] = param_group.attrs[key]

        return spectrograms, params
    
    def _create_splits(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create train/validation splits for both genres. Test set is implied as the rest.
        """
        print("Creating train/validation splits...")
        
        splits = {'train': {}, 'val': {}}
        
        for genre_name, spectrograms in [('A', self.genre_a_data), ('B', self.genre_b_data)]:
            # Split data into training and validation sets
            train_data, val_data = train_test_split(
                spectrograms,
                test_size=self.val_split,
                random_state=self.seed,
                shuffle=True
            )
            
            splits['train'][genre_name] = train_data
            splits['val'][genre_name] = val_data
            
            print(f"Genre {genre_name} - Train: {len(train_data)}, Val: {len(val_data)}")
        
        return splits
    
    def _preprocess_spectrogram(self, spectrogram):
        """
        Prepares spectrogram for training.
        The data is already in the correct (H, W, 2) shape.
        """
        
        # Ensure float32 type
        spectrogram = tf.cast(spectrogram, tf.float32)
        
        return spectrogram
    
    def create_tf_dataset(self, spectrograms: np.ndarray, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from spectrograms 
        """
        dataset = tf.data.Dataset.from_tensor_slices(spectrograms)
        dataset = dataset.map(self._preprocess_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=self.seed)
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.prefetch_size)
        
        return dataset
    
    def get_datasets(self) -> Dict[str, tf.data.Dataset]:
        """
        Get TensorFlow datasets for training
        """
        datasets = {}
        for split in ['train', 'val']:
            shuffle = (split == 'train')
            
            datasets[f'{split}_A'] = self.create_tf_dataset(self.splits[split]['A'], shuffle=shuffle)
            datasets[f'{split}_B'] = self.create_tf_dataset(self.splits[split]['B'], shuffle=shuffle)
        
        return datasets
    
    def visualize_samples(self, num_samples: int = 4, split: str = 'train'):
        """
        Visualize sample spectrograms, showing both Mel and CQT channels.
        """
        # We need 4 rows: Genre A Mel, Genre A CQT, Genre B Mel, Genre B CQT
        fig, axes = plt.subplots(4, num_samples, figsize=(15, 10))
        fig.suptitle(f'Sample Spectrograms from "{split}" set', fontsize=16)

        for i in range(num_samples):
            # --- Genre A ---
            idx_a = np.random.randint(0, len(self.splits[split]['A']))
            spec_a = self.splits[split]['A'][idx_a]
            
            # Mel channel (channel 0)
            mel_a = spec_a[:, :, 0]
            axes[0, i].imshow(mel_a, aspect='auto', origin='lower', cmap='viridis')
            axes[0, i].set_title(f'Genre A Mel #{i+1}')
            axes[0, i].axis('off')
            
            # CQT channel (channel 1)
            cqt_a = spec_a[:, :, 1]
            axes[1, i].imshow(cqt_a, aspect='auto', origin='lower', cmap='magma')
            axes[1, i].set_title(f'Genre A CQT #{i+1}')
            axes[1, i].axis('off')

            # --- Genre B ---
            idx_b = np.random.randint(0, len(self.splits[split]['B']))
            spec_b = self.splits[split]['B'][idx_b]
            
            # Mel channel (channel 0)
            mel_b = spec_b[:, :, 0]
            axes[2, i].imshow(mel_b, aspect='auto', origin='lower', cmap='viridis')
            axes[2, i].set_title(f'Genre B Mel #{i+1}')
            axes[2, i].axis('off')
            
            # CQT channel (channel 1)
            cqt_b = spec_b[:, :, 1]
            axes[3, i].imshow(cqt_b, aspect='auto', origin='lower', cmap='magma')
            axes[3, i].set_title(f'Genre B CQT #{i+1}')
            axes[3, i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()