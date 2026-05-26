import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .encoder import KeystrokeEncoder
from .dataset import KeystrokeWindowDataset, collate_windows


# =============================================================
# CONFIG
# =============================================================

WINDOW_SIZE  = 20
STRIDE       = 10
EMBED_DIM    = 64
NUM_LAYERS   = 2
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 3e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================
# CONTRASTIVE LOSS  (NT-Xent / InfoNCE style)
# Each sample in the batch is its own class — same as teammate's Phase 2
# =============================================================

def contrastive_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Self-supervised contrastive loss.
    Encourages each embedding to be most similar to itself
    and dissimilar to all others in the batch.

    z : (B, embed_dim)
    """
    z   = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)                           # (B, B)
    labels = torch.arange(len(z), device=z.device)
    return F.cross_entropy(sim, labels)


def contrastive_accuracy(z: torch.Tensor) -> float:
    z   = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)
    preds = sim.argmax(dim=1)
    labels = torch.arange(len(z), device=z.device)
    return (preds == labels).float().mean().item()


# =============================================================
# TRAIN
# =============================================================

def train(
    events:        list[dict],
    bidirectional: bool  = False,
    window_size:   int   = WINDOW_SIZE,
    stride:        int   = STRIDE,
    embed_dim:     int   = EMBED_DIM,
    num_layers:    int   = NUM_LAYERS,
    batch_size:    int   = BATCH_SIZE,
    epochs:        int   = EPOCHS,
    lr:            float = LR,
    val_split:     float = 0.1,
) -> KeystrokeEncoder:
    """
    Train the keystroke encoder using contrastive loss on a list of
    preprocessed keystroke event dicts.

    Parameters
    ----------
    events        : output of parse_csv_events() — list of {code, hold, ikl}
    bidirectional : False = LSTM, True = BiLSTM
    ...

    Returns
    -------
    Trained KeystrokeEncoder (on CPU, eval mode)
    """
    print(f"Device : {DEVICE}")
    print(f"Variant: {'BiLSTM' if bidirectional else 'LSTM'}")
    print(f"Events : {len(events)}  →  windows: {max(0, (len(events) - window_size) // stride + 1)}")

    dataset = KeystrokeWindowDataset(events, window_size, stride)

    if len(dataset) < 2:
        raise ValueError(
            f"Not enough events to form windows. "
            f"Need at least {window_size} events, got {len(events)}."
        )

    # train / val split
    val_n   = max(1, int(len(dataset) * val_split))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_windows,
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_windows,
        num_workers = 0,
    )

    model = KeystrokeEncoder(
        input_size    = 3,
        hidden_size   = embed_dim,
        num_layers    = num_layers,
        bidirectional = bidirectional,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):

        # ---- train ----
        model.train()
        t_loss = 0.0
        t_acc  = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            batch = batch.to(DEVICE)          # (B, W, 3)
            z     = model(batch)              # (B, embed_dim)
            loss  = contrastive_loss(z)
            acc   = contrastive_accuracy(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_acc  += acc

        # ---- val ----
        model.eval()
        v_loss = 0.0
        v_acc  = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(DEVICE)
                z      = model(batch)
                v_loss += contrastive_loss(z).item()
                v_acc  += contrastive_accuracy(z)

        print(
            f"  train loss: {t_loss/len(train_loader):.4f}  "
            f"acc: {t_acc/len(train_loader):.4f}  |  "
            f"val loss: {v_loss/len(val_loader):.4f}  "
            f"acc: {v_acc/len(val_loader):.4f}"
        )

    model.eval().cpu()
    return model


# =============================================================
# EXTRACT EMBEDDINGS  (offline / batch)
# =============================================================

def extract_embeddings(
    model:       KeystrokeEncoder,
    events:      list[dict],
    window_size: int = WINDOW_SIZE,
    stride:      int = STRIDE,
    batch_size:  int = BATCH_SIZE,
) -> np.ndarray:
    """
    Run the trained encoder over all windows from an event list.

    Returns
    -------
    np.ndarray  shape (N_windows, embed_dim)  dtype float32
    """
    dataset = KeystrokeWindowDataset(events, window_size, stride)
    loader  = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = False,
        collate_fn = collate_windows,
    )

    model.eval()
    device = next(model.parameters()).device
    out    = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z     = model(batch)
            out.append(z.cpu().numpy())

    return np.vstack(out).astype(np.float32)