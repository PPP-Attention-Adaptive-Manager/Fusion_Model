"""
train_keyboard_encoder.py
Run from ~/fusion_model/ — mirrors prepare_tucker_inputs.py session discovery.
"""
import argparse, logging, sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pre_embedders.keyboard import KeystrokeEncoder, parse_csv_events, KeystrokeWindowDataset
from pre_embedders.keyboard.dataset import collate_windows
from pre_embedders.keyboard.train import contrastive_loss, contrastive_accuracy

def find_sessions(user_dir):
    results = []
    for p in sorted(user_dir.rglob("session_*")):
        if not p.is_dir(): continue
        rel_parts = p.relative_to(user_dir).parts
        if any(part.startswith("session_") for part in rel_parts[:-1]): continue
        results.append(p)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     type=str, default=str(Path.home() / "aam_data" / "Data" / "After update"))
    parser.add_argument("--output_dir",    type=str, default=str(Path(__file__).resolve().parent / "pre_embedders" / "keyboard" / "weights"))
    parser.add_argument("--users",         nargs="*", default=None)
    parser.add_argument("--bidirectional", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--window_size",   type=int, default=20)
    parser.add_argument("--stride",        type=int, default=10)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--lr",            type=float, default=3e-4)
    args = parser.parse_args()

    data_root  = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    user_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if args.users:
        user_dirs = [d for d in user_dirs if d.name in args.users]
    log.info("users found: %d", len(user_dirs))

    session_datasets = []
    total_windows = 0

    for user_dir in user_dirs:
        for session_path in find_sessions(user_dir):
            kb_csv = session_path / "raw" / "keyboard.csv"
            if not kb_csv.exists(): continue
            try:
                rows   = pd.read_csv(kb_csv).to_dict("records")
                events = parse_csv_events(rows)
                if len(events) < args.window_size: continue
                ds = KeystrokeWindowDataset(events, args.window_size, args.stride)
                if len(ds) == 0: continue
                session_datasets.append(ds)
                total_windows += len(ds)
                log.info("  %s / %s  →  %d events  %d windows", user_dir.name, session_path.name, len(events), len(ds))
            except Exception as e:
                log.debug("  skip %s: %s", session_path, e)

    log.info("total sessions: %d  total windows: %d", len(session_datasets), total_windows)

    if not session_datasets:
        log.error("No keyboard sessions found. Check --data_root.")
        sys.exit(1)

    batch_size = min(args.batch_size, total_windows // 2)
    batch_size = max(2, batch_size)

    loader = DataLoader(ConcatDataset(session_datasets), batch_size=batch_size, shuffle=True, collate_fn=collate_windows)

    model = KeystrokeEncoder(input_size=3, hidden_size=64, num_layers=2, bidirectional=args.bidirectional).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        t_loss = t_acc = 0.0
        for batch in loader:
            batch = batch.to(device)
            z     = model(batch)
            loss  = contrastive_loss(z)
            acc   = contrastive_accuracy(z)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item(); t_acc += acc
        log.info("Epoch %2d/%d  loss: %.4f  acc: %.4f", epoch+1, args.epochs, t_loss/len(loader), t_acc/len(loader))

    variant   = "bilstm" if args.bidirectional else "lstm"
    save_path = output_dir / f"kb_encoder_{variant}.pt"
    model.eval().cpu()
    torch.save(model.state_dict(), save_path)
    log.info("weights saved → %s", save_path)

if __name__ == "__main__":
    main()
