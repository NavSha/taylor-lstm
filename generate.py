import argparse
import torch
from model import CharLSTM
from preprocess import LyricsDataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path, vocab_path, device):
    char_to_idx, idx_to_char = LyricsDataset.load_vocab(vocab_path)
    vocab_size = len(char_to_idx)

    model = CharLSTM(vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, char_to_idx, idx_to_char


def generate(model, seed_text, char_to_idx, idx_to_char, device, length=500, temperature=0.8):
    # Encode seed
    chars = [ch for ch in seed_text if ch in char_to_idx]
    if not chars:
        print("Warning: seed text has no characters in vocabulary, using first vocab char.")
        chars = [idx_to_char[0]]

    input_seq = torch.tensor([[char_to_idx[ch] for ch in chars]], dtype=torch.long, device=device)

    # Prime the hidden state with seed
    hidden = None
    with torch.no_grad():
        logits, hidden = model(input_seq, hidden)

    # Start generating from last character's prediction
    result = list(seed_text)
    last_logits = logits[0, -1]  # logits for the last position

    for _ in range(length):
        # Apply temperature
        scaled = last_logits / temperature
        probs = torch.softmax(scaled, dim=0)
        idx = torch.multinomial(probs, 1).item()

        result.append(idx_to_char[idx])

        # Feed back
        next_input = torch.tensor([[idx]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, hidden = model(next_input, hidden)
        last_logits = logits[0, -1]

    return "".join(result)


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    import os
    import glob
    files = glob.glob(os.path.join(checkpoint_dir, "model_epoch*.pt"))
    if not files:
        return None
    # Sort by epoch number
    files.sort(key=lambda f: int(f.split("epoch")[1].split(".")[0]))
    return files[-1]


def interactive_mode(model, char_to_idx, idx_to_char, device):
    print("\n=== Taylor Swift Lyrics Generator ===")
    print("Type a seed phrase and press Enter. Type 'quit' to exit.\n")

    while True:
        seed = input("Seed text: ").strip()
        if seed.lower() == "quit":
            break

        try:
            temp = float(input("Temperature (0.2-1.5, default 0.8): ").strip() or "0.8")
        except ValueError:
            temp = 0.8

        try:
            length = int(input("Length (default 500): ").strip() or "500")
        except ValueError:
            length = 500

        print("\n--- Generated Lyrics ---")
        text = generate(model, seed, char_to_idx, idx_to_char, device, length=length, temperature=temp)
        print(text)
        print("--- End ---\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Taylor Swift-style lyrics")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json")
    parser.add_argument("--seed", type=str, default=None, help="Seed text (interactive if omitted)")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if not checkpoint_path:
        print("No checkpoint found in checkpoints/. Train the model first with: python train.py")
        return

    print(f"Loading model from: {checkpoint_path}")
    model, char_to_idx, idx_to_char = load_model(checkpoint_path, args.vocab, device)

    if args.seed:
        text = generate(model, args.seed, char_to_idx, idx_to_char, device,
                        length=args.length, temperature=args.temp)
        print(text)
    else:
        interactive_mode(model, char_to_idx, idx_to_char, device)


if __name__ == "__main__":
    main()
