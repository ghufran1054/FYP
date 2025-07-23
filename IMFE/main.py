from src.imfe import IMFE, BatchedIMFE
from dataset.dataloader import RGBFrameFeatureDataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os




# # ===== STRICT CPU LIMITATION =====
# os.environ["OMP_NUM_THREADS"] = "8"         # OpenMP threads
# os.environ["MKL_NUM_THREADS"] = "8"         # Intel MKL
# os.environ["OPENBLAS_NUM_THREADS"] = "8"    # OpenBLAS
# os.environ["NUMEXPR_NUM_THREADS"] = "8"     # NumExpr
# os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Accelerate framework on macOS
# # Now set PyTorch specific limits
# torch.set_num_threads(8)  # Main thread pool
# torch.set_num_interop_threads(8)  # Inter-operation threads
# Initialize the dataset and dataloader
rgb_root = '/data1/ghufran/validation_frames'
feature_root = '/data1/ghufran/helping_python_scripts/IMFE_train'
dataset_train = RGBFrameFeatureDataset(rgb_root, feature_root)
rgb_root_val = '/data1/ghufran/test_frames'
feature_root_val = '/data1/ghufran/helping_python_scripts/IMFE_val'
dataset_val = RGBFrameFeatureDataset(rgb_root_val, feature_root_val)

print(f"Training dataset size: {len(dataset_train)}")
print(f"Validation dataset size: {len(dataset_val)}")

# model = IMFE()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batched_model = BatchedIMFE()
# exit()

from torch.optim.lr_scheduler import LambdaLR





def train_imfe(model, train_dataset, val_dataset=None, save_path="best_imfe.pth", epochs=30, lr=1e-3, batch_size=8, device='cuda'):

    model = model.to(device)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) if val_dataset else None

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_loss = float('inf')
    def lr_lambda(step):
        return 0.9 ** (step // decay_every_n_steps)

    iters_per_epoch = len(dataloader)  # e.g. 2522
    decay_every_n_steps = iters_per_epoch // 5

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for frames, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            frames, targets = frames.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(frames)  # [B, 2048]
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()    

            running_loss += loss.item() * frames.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.6f}")
        # Validate
        if valloader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for frames, targets in tqdm(valloader, desc="Validation"):
                    frames, targets = frames.to(device), targets.to(device)
                    preds = model(frames)
                    pred_mean = preds.mean().item()
                    pred_std = preds.std().item()
                    has_nan = torch.isnan(preds).any().item()
                    has_inf = torch.isinf(preds).any().item()
                    if pred_std > 100 or has_nan or has_inf:
                        print(f"Warning: Unstable predictions detected! Mean: {pred_mean:.4f}, Std: {pred_std:.4f}, NaN: {has_nan}, Inf: {has_inf}")
                    val_loss += criterion(preds, targets).item() * frames.size(0)

            val_loss /= len(val_dataset)
            print(f"Val Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_path)

                # Also save all the current training parameters such as learning rate, epoch, etc. to resume later
                with open(save_path.replace('.pth', '.txt'), 'w') as f:
                    f.write(f"Epoch: {epoch+1}, Best Val Loss: {best_loss:.6f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")

                print("Saved new best model!")

        else:
            torch.save(model.state_dict(), save_path)


train_imfe(batched_model, dataset_train, val_dataset=dataset_val, save_path="best_imfe.pth", epochs=20, lr=1e-4, batch_size=32, device='cuda')