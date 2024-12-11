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
        # GPU-specific checks
        assert not (USE_AMP and not torch.cuda.is_available())

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

        # Optimizer setup
        init_model(params)
        from ngpt.model import model

        self.model = model

        self.optim = Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Register normalizing step if not using parameterization
        if not USE_PARAMETRIZE:
            model.register_step_post_hook(self.optim)

        # Training parameters
        self.BATCHES_PER_EPOCH = len(self.train_dataset) // BATCH_SIZE
        self.STEPS_PER_EPOCH = len(self.train_dataset) // BATCH_SIZE
        self.training_tokens = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self):
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

                self.model.train()
                data = data.to(device)

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=USE_AMP
                ):
                    loss = self.model(data, return_loss=True)

                loss = loss / GRAD_ACCUM_EVERY
                loss.backward()

                running_loss += loss.item()

                # Gradient accumulation step
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

                tokens_seen_so_far += data.numel()
                self.training_tokens.append(tokens_seen_so_far)

            # Validation step
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

            construct_metrics()

            # Save checkpoint at each epoch
            print(f"Saving model at epoch {epoch + 1}")
            torch.save(self.model.state_dict(), EPOCH_MODEL)

        construct_metrics()
        torch.save(self.model.state_dict(), TRAIN_MODEL)

        train_end_time = time.time()
        time_training = train_end_time - train_start_time
        print(f"Training Time: {time_training:.3f} seconds")
