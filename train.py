import os
import torch
import torch.nn as nn
from preprocess import get_dataloader
from model import CharLSTM

# --- Hyperparameters ---
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
EPOCHS = 50
PRINT_EVERY = 200  # batches
SAVE_EVERY = 10    # epochs

DATA_PATH = "data/lyrics.txt"
CHECKPOINT_DIR = "checkpoints"
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train():
    device = get_device()
    print(f"Using device: {device}")

    # Data
    dataset, loader = get_dataloader(DATA_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    print(f"Vocab size: {dataset.vocab_size}, Dataset size: {len(dataset):,} sequences")

    # Save vocab for generation
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dataset.save_vocab(VOCAB_PATH)

    # Model
    model = CharLSTM(
        vocab_size=dataset.vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss = float("inf")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            # logits: (batch, seq_len, vocab_size), y: (batch, seq_len)
            loss = criterion(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % PRINT_EVERY == 0:
                avg = total_loss / batch_count
                print(f"  Epoch {epoch}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f} (avg: {avg:.4f})")

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch}/{EPOCHS} — Avg Loss: {avg_loss:.4f} (lr: {optimizer.param_groups[0]['lr']:.6f})")
        scheduler.step(avg_loss)

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CHECKPOINT_DIR, "model_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": dataset.vocab_size,
                "loss": avg_loss,
            }, best_path)
            print(f"  New best model! Saved to {best_path}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": dataset.vocab_size,
                "loss": avg_loss,
            }, path)
            print(f"  Saved checkpoint: {path}")

    print("Training complete!")


if __name__ == "__main__":
    train()
