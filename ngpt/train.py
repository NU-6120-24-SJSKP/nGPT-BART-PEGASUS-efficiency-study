import time

import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from ngpt.config import *
from ngpt.data import dataset, TextSamplerDataset, collate_fn
from ngpt.helpers import *
from ngpt.metrics import construct_metrics
from ngpt.model import USE_PARAMETRIZE, init_model
from ngpt.test import generate_summary
from ngpt.validate import validate_model


class NGPTTrainer:
    def __init__(self, params):
        # Ensure Automatic Mixed Precision (AMP) is only used with CUDA
        assert not (USE_AMP and not torch.cuda.is_available()), "AMP requires CUDA"

        # Prepare datasets and dataloaders
        self.train_dataset = TextSamplerDataset(dataset["train"], SEQ_LEN)
        self.train_loader = cycle(
            DataLoader(
                self.train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
            )
        )

        # Initialize the model and optimizer
        init_model(params)
        from ngpt.model import model

        self.model = model
        self.optim = Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Register a step hook for normalization if not using parameterization
        if not USE_PARAMETRIZE:
            model.register_step_post_hook(self.optim)

        # Training parameters
        self.BATCHES_PER_EPOCH = len(self.train_dataset) // BATCH_SIZE
        self.STEPS_PER_EPOCH = len(self.train_dataset) // BATCH_SIZE
        self.training_tokens = []  # To track the number of tokens seen during training
        self.best_val_loss = float("inf")  # Initialize best validation loss
        self.patience_counter = 0  # Counter for early stopping

    def train(self):
        """
        Main training loop for the NGPT model.
        """
        train_start_time = time.time()
        epoch_iterator = tqdm.tqdm(range(NUM_EPOCHS), mininterval=10.0, desc="training")

        tokens_seen_so_far = 0
        for epoch in epoch_iterator:
            epoch_start_time = time.time()
            batch_iterator = tqdm.tqdm(
                enumerate(self.train_loader),
                total=self.STEPS_PER_EPOCH,
                desc=f"Epoch {epoch + 1}",
                leave=False,
            )
            train_loss_per_batch = []

            running_loss = 0.0
            for batch_idx, data in batch_iterator:
                if batch_idx >= self.STEPS_PER_EPOCH:
                    break

                # Set model to training mode
                self.model.train()
                data = data.to(device)

                # Use Automatic Mixed Precision if enabled
                with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=USE_AMP
                ):
                    loss = self.model(data, return_loss=True)

                # Normalize loss for gradient accumulation
                loss = loss / GRAD_ACCUM_EVERY
                loss.backward()

                running_loss += loss.item()

                # Perform optimization step after accumulating gradients
                if (batch_idx + 1) % GRAD_ACCUM_EVERY == 0 or (
                        batch_idx + 1 == self.STEPS_PER_EPOCH
                ):
                    scaler.step(self.optim)
                    scaler.update()
                    self.optim.zero_grad()

                    curr_loss = running_loss / GRAD_ACCUM_EVERY
                    train_loss_per_batch.append(curr_loss)
                    print(f"Training loss: {curr_loss:.3f}")

                    running_loss = 0.0

                # Track tokens seen during training
                tokens_seen_so_far += data.numel()
                self.training_tokens.append(tokens_seen_so_far)

            # Validation step after each epoch
            if validate_model(epoch) == "stop":
                break

            # Generate summaries after each epoch
            generate_summary(epoch)

            # Log average training loss for the epoch
            epoch_loss = sum(train_loss_per_batch) / len(train_loss_per_batch)
            print(f"Epoch {epoch + 1}, Average Loss over the epoch: {epoch_loss:.4f}")

            epoch_end_time = time.time()
            time_per_epoch = (epoch_end_time - epoch_start_time) / (epoch + 1)
            print(f"Time per Epoch: {time_per_epoch:.4f} seconds")

            # Construct metrics for logging
            construct_metrics()

            # Save checkpoint at each epoch
            print(f"Saving model at epoch {epoch + 1}")
            torch.save(self.model.state_dict(), EPOCH_MODEL)

        # Final metrics construction and model saving
        construct_metrics()
        torch.save(self.model.state_dict(), TRAIN_MODEL)

        train_end_time = time.time()
        time_training = train_end_time - train_start_time
        print(f"Training Time: {time_training:.3f} seconds")
