import tensorflow as tf
import numpy as np
from tensorflow.keras import losses, optimizers
import matplotlib.pyplot as plt

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler that implements linear decay.
    
    Initializes with a constant learning rate for a certain number of steps
    (decay_start_step), then linearly decays to zero over the remaining steps.
    """
    def __init__(self, initial_learning_rate, total_steps, decay_start_step):
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.total_steps = float(total_steps)
        self.decay_start_step = float(decay_start_step)

    def __call__(self, step):
        # Cast step to float for calculations
        step = tf.cast(step, tf.float32)
        
        # Define the functions for the true and false branches of our condition
        def constant_lr_fn():
            # This is executed if step < decay_start_step
            return self.initial_learning_rate

        def decayed_lr_fn():
            # This is executed if step >= decay_start_step
            decay_range = self.total_steps - self.decay_start_step
            steps_into_decay = step - self.decay_start_step
            
            decay_factor = 1.0 - (steps_into_decay / decay_range)
            
            return self.initial_learning_rate * tf.maximum(0.0, decay_factor)

        
        # builds the conditional logic into the TensorFlow graph
        return tf.cond(
            pred=step < self.decay_start_step,
            true_fn=constant_lr_fn,
            false_fn=decayed_lr_fn
        )


class CycleGAN:
    def __init__(self, generator_AB, generator_BA, discriminator_A, discriminator_B, 
                 lambda_cycle=10.0, lambda_identity_start=0.5, lambda_identity_end=0.0,
                 identity_decay_epochs=50, initial_gen_lr=2e-4, initial_disc_lr=1e-4,
                 total_train_steps=None,lr_decay_start_step=None):
        """
        Initialize CycleGAN for music genre translation
        
        Args:
            generator_AB: Generator that translates from genre A to B
            generator_BA: Generator that translates from genre B to A
            discriminator_A: Discriminator for genre A
            discriminator_B: Discriminator for genre B
            lambda_cycle: Weight for cycle consistency loss
            lambda_identity: Weight for identity loss
        """
        self.gen_AB = generator_AB
        self.gen_BA = generator_BA
        self.disc_A = discriminator_A
        self.disc_B = discriminator_B
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity_start = lambda_identity_start
        self.lambda_identity_end = lambda_identity_end
        self.identity_decay_epochs = identity_decay_epochs
        self.current_lambda_identity = lambda_identity_start
       
        # Optimizers with Learning Rate Schedulers
        if total_train_steps is None or lr_decay_start_step is None:
            raise ValueError("total_train_steps and lr_decay_start_step must be provided for LR scheduling.")

        # Create a learning rate schedule for the generator
        gen_lr_schedule = LinearDecay(
            initial_learning_rate=initial_gen_lr,
            total_steps=total_train_steps,
            decay_start_step=lr_decay_start_step
        )
        
        # Create a learning rate schedule for the discriminator
        disc_lr_schedule = LinearDecay(
            initial_learning_rate=initial_disc_lr,
            total_steps=total_train_steps,
            decay_start_step=lr_decay_start_step
        )
        
        # Pass the schedule objects to the optimizers
        self.gen_optimizer = optimizers.Adam(learning_rate=gen_lr_schedule, beta_1=0.5)
        self.disc_optimizer = optimizers.Adam(learning_rate=disc_lr_schedule, beta_1=0.5)
        
        # Loss tracking
        self.gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')
        self.cycle_loss_tracker = tf.keras.metrics.Mean(name='cycle_loss')
        self.identity_loss_tracker = tf.keras.metrics.Mean(name='identity_loss')
        self.epoch_gen_losses = []
        self.epoch_disc_losses = []
        self.epoch_cycle_losses = []
        self.epoch_identity_losses = []



    def update_lambda_identity(self, current_epoch):
        """
        Update lambda_identity based on current epoch using linear decay
        """
        if current_epoch < self.identity_decay_epochs:
            # Linear decay from start to end value
            decay_factor = current_epoch / self.identity_decay_epochs
            self.current_lambda_identity = (
                self.lambda_identity_start * (1 - decay_factor) + 
                self.lambda_identity_end * decay_factor
            )
        else:
            # After decay period, use end value
            self.current_lambda_identity = self.lambda_identity_end
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Calculate discriminator loss using least squares GAN loss
        """
        real_loss = tf.reduce_mean(tf.square(real_output - 0.9))#label smoothing
        fake_loss = tf.reduce_mean(tf.square(fake_output))
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, fake_output):
        """
        Calculate generator loss using least squares GAN loss
        """
        return tf.reduce_mean(tf.square(fake_output - 1))
    
    def cycle_consistency_loss(self, real_image, cycled_image):
        """
        Calculate cycle consistency loss (L1 loss)
        """
        return tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    def identity_loss(self, real_image, same_image):
        """
        Calculate identity loss (L1 loss)
        """
        return tf.reduce_mean(tf.abs(real_image - same_image))
 
    @tf.function
    def train_step(self, real_A, real_B):
        """
        #Single training step for CycleGAN
        """
        
        with tf.GradientTape(persistent= True) as tape:
            # Generate fake images
            fake_B = self.gen_AB(real_A, training=True)
            fake_A = self.gen_BA(real_B, training=True)

            fake_B_mel_only = fake_B[..., 0:1] 
            fake_A_mel_only = fake_A[..., 0:1]
            
            
            # Cycle the fake images back
            cycled_A = self.gen_BA(fake_B, training=True)
            cycled_B = self.gen_AB(fake_A, training=True)
            
            # Identity mapping (helps preserve color/timbre)
            same_A = self.gen_BA(real_A, training=True)
            same_B = self.gen_AB(real_B, training=True)
            

            # Slice to keep the channel dim: (B, H, W, 1)
            real_A_mel_only = real_A[..., 0:1] 
            real_B_mel_only = real_B[..., 0:1]
            

            #Create Gaussian noise with the same shape as the inputs.
            # This will be added to the REAL spectrograms and Q transform before feeding them to the discriminator.
            noise_A = tf.random.normal(tf.shape(real_A_mel_only), stddev=0.02)
            noise_B = tf.random.normal(tf.shape(real_B_mel_only), stddev=0.02)


            # Discriminator outputs
            disc_real_A = self.disc_A(real_A_mel_only + noise_A, training=True)
            disc_real_B = self.disc_B(real_B_mel_only + noise_B, training=True)
            disc_fake_A = self.disc_A(fake_A_mel_only, training=True)
            disc_fake_B = self.disc_B(fake_B_mel_only, training=True)
            
            # Generator losses
            gen_AB_loss = self.generator_loss(disc_fake_B)
            gen_BA_loss = self.generator_loss(disc_fake_A)
            
            # Cycle consistency losses
            cycle_loss_A = self.cycle_consistency_loss(real_A, cycled_A)
            cycle_loss_B = self.cycle_consistency_loss(real_B, cycled_B)
            cycle_loss = cycle_loss_A + cycle_loss_B
            
            # Identity losses
            identity_loss_A = self.identity_loss(real_A, same_A)
            identity_loss_B = self.identity_loss(real_B, same_B)
            identity_loss = identity_loss_A + identity_loss_B
            
            # Total generator loss
            total_gen_loss = (gen_AB_loss + gen_BA_loss + 
                            self.lambda_cycle * cycle_loss + 
                            self.current_lambda_identity * identity_loss)
            
            # Discriminator losses
            disc_A_loss = self.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss(disc_real_B, disc_fake_B)
            total_disc_loss = disc_A_loss + disc_B_loss
        
        # Calculate gradients
        gen_gradients = tape.gradient(total_gen_loss, 
                                    self.gen_AB.trainable_variables + 
                                    self.gen_BA.trainable_variables)
        
        disc_gradients = tape.gradient(total_disc_loss,
                                     self.disc_A.trainable_variables + 
                                     self.disc_B.trainable_variables)
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_gradients, 
                                             self.gen_AB.trainable_variables + 
                                             self.gen_BA.trainable_variables))
        
        self.disc_optimizer.apply_gradients(zip(disc_gradients,
                                              self.disc_A.trainable_variables + 
                                              self.disc_B.trainable_variables))
        
        # Update metrics
        self.gen_loss_tracker.update_state(total_gen_loss)
        self.disc_loss_tracker.update_state(total_disc_loss)
        self.cycle_loss_tracker.update_state(cycle_loss)
        self.identity_loss_tracker.update_state(identity_loss)

        
        
        return {
            'gen_loss': self.gen_loss_tracker.result(),
            'disc_loss': self.disc_loss_tracker.result(),
            'cycle_loss': self.cycle_loss_tracker.result(),
            'identity_loss': self.identity_loss_tracker.result()
        }

    
    def train(self, dataset_A, dataset_B, epochs, steps_per_epoch, 
              checkpoint_dir=None, sample_dir=None):
        """
        Train the CycleGAN
        
        Args:
            dataset_A: tf.data.Dataset for genre A mel spectrograms and constant Q transform
            dataset_B: tf.data.Dataset for genre B mel spectrograms and constant Q transform
            epochs: Number of training epochs
            steps_per_epoch: Number of steps per epoch
            checkpoint_dir: Directory to save checkpoints
            sample_dir: Directory to save sample translations
        """
        # Setup checkpointing
        if checkpoint_dir:
            checkpoint = tf.train.Checkpoint(
                gen_AB=self.gen_AB,
                gen_BA=self.gen_BA,
                disc_A=self.disc_A,
                disc_B=self.disc_B,
                gen_optimizer=self.gen_optimizer,
                disc_optimizer=self.disc_optimizer
            )
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, checkpoint_dir, max_to_keep=5
            )
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            

            # Update lambda_identity for this epoch
            self.update_lambda_identity(epoch)
            
            # Reset metrics
            self.gen_loss_tracker.reset_states()
            self.disc_loss_tracker.reset_states()
            self.cycle_loss_tracker.reset_states()
            self.identity_loss_tracker.reset_states()
            
            # Zip datasets for paired training
            paired_dataset = tf.data.Dataset.zip((dataset_A, dataset_B))
            
            for step, (real_A, real_B) in enumerate(paired_dataset.take(steps_per_epoch)):
                losses = self.train_step(real_A, real_B)
                
                # Print progress
                if step % 100 == 0:
                    print(f"Step {step}: Gen Loss: {losses['gen_loss']:.4f}, "
                          f"Disc Loss: {losses['disc_loss']:.4f}, "
                          f"Cycle Loss: {losses['cycle_loss']:.4f},"
                          f"Identity Loss: {losses['identity_loss']:.4f} ")
            
            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                checkpoint_manager.save()
                print(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Generate sample translations
            if sample_dir and (epoch + 1) % 5 == 0:
                self.generate_samples(dataset_A, dataset_B, epoch + 1, sample_dir)
            
            self.epoch_gen_losses.append(self.gen_loss_tracker.result().numpy())
            self.epoch_disc_losses.append(self.disc_loss_tracker.result().numpy())
            self.epoch_cycle_losses.append(self.cycle_loss_tracker.result().numpy())
            self.epoch_identity_losses.append(self.identity_loss_tracker.result().numpy())
    
    def generate_samples(self, dataset_A, dataset_B, epoch, save_dir):
        """
        Generate and save sample translations
        """
        # Take a few samples from each dataset
        sample_A = next(iter(dataset_A.take(1)))
        sample_B = next(iter(dataset_B.take(1)))

        sample_A_mel = sample_A[..., 0:1] 
        sample_B_mel = sample_B[..., 0:1] 
        
        # Generate translations
        fake_B = self.gen_AB(sample_A, training=False)
        fake_A = self.gen_BA(sample_B, training=False)

        fake_B_mel = fake_B[..., 0:1] 
        fake_A_mel = fake_A[..., 0:1]
        
        # Cycle back
        cycled_A = self.gen_BA(fake_B, training=False)
        cycled_B = self.gen_AB(fake_A, training=False)

        cycled_A_mel = cycled_A[..., 0:1] 
        cycled_B_mel = cycled_B[..., 0:1]
        
        # Plot and save
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # First row: A -> B -> A
        axes[0, 0].imshow(sample_A_mel[0].numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Real A')
        axes[0, 1].imshow(fake_B_mel[0].numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Fake B')
        axes[0, 2].imshow(cycled_A_mel[0].numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[0, 2].set_title('Cycled A')
        axes[0, 3].imshow((sample_A_mel[0] - cycled_A_mel[0]).numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[0, 3].set_title('Difference')
        
        # Second row: B -> A -> B
        axes[1, 0].imshow(sample_B_mel[0].numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Real B')
        axes[1, 1].imshow(fake_A_mel[0].numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Fake A')
        axes[1, 2].imshow(cycled_B_mel[0].numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[1, 2].set_title('Cycled B')
        axes[1, 3].imshow((sample_B_mel[0] - cycled_B_mel[0]).numpy().squeeze(), aspect='auto', cmap='viridis')
        axes[1, 3].set_title('Difference')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/samples_epoch_{epoch}.png")
        plt.close()
    
    def plot_training_losses(self, save_path=None):
        """
        Plot training losses stored during training.

        Args:
            save_path: Optional file path to save the figure instead of showing it.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(self.epoch_gen_losses, label='Generator Loss')
        plt.plot(self.epoch_disc_losses, label='Discriminator Loss')
        plt.plot(self.epoch_cycle_losses, label='Cycle Consistency Loss')
        plt.plot(self.epoch_identity_losses, label='Identity Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CycleGAN Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to: {save_path}")
        else:
            plt.show()
