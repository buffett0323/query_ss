import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dismix_model import DisMixModel
from dataset import CocoChoralesTinyDataset
from dismix_loss import ELBOLoss, BarlowTwinsLoss

# Initialize wandb
wandb.init(project="disMix-training", config={
    "batch_size": 32,
    "learning_rate": 4e-4,
    "epochs": 10000000
})

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 4e-4
clip_value = 0.5
early_stop_patience = 260000
best_val_loss = float('inf')
np_improvement_steps = 0
max_steps = 10000000

# Usage Example
data_dir = '/home/buffett/NAS_189/cocochorales_full_v1_output/main_dataset/'
train_dataset = CocoChoralesTinyDataset(data_dir, split='train')
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_data_loader_iter = iter(train_data_loader)

valid_dataset = CocoChoralesTinyDataset(data_dir, split='valid')
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CocoChoralesTinyDataset(data_dir, split='test')
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Models and other settings
model = DisMixModel(
    input_dim=128, 
    latent_dim=64, 
    hidden_dim=256, 
    gru_hidden_dim=256,
    num_frames=10,
    pitch_classes=52,
    output_dim=128,    
).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Loss function
elbo_loss_fn = ELBOLoss() # For ELBO
bce_loss_fn = nn.BCEWithLogitsLoss()  # For pitch supervision
bt_loss_fn = BarlowTwinsLoss() # Barlow Twins

# Log model and hyperparameters in wandb
wandb.watch(model, log="all", log_freq=10)

# Training
for step in tqdm(range(max_steps)):
    model.train()
    optimizer.zero_grad()
    
    # Call next() on the iterator
    try:
        batch = next(train_data_loader_iter)
    except StopIteration:
        train_data_loader_iter = iter(train_data_loader)  # Reset iterator
        batch = next(train_data_loader_iter)
        
    mixture = batch["mixture"].to(device)
    query = batch["query"].to(device)
    pitch_annotation = batch["pitch_annotation"].to(device) # pitch_label = batch["pitch_label"].to(device)
        
    rec_mixture, pitch_latent, pitch_logits, timbre_latent, timbre_mean, timbre_logvar, eq = model(mixture, query)
    
    # Loss
    elbo_loss = elbo_loss_fn(
        mixture, rec_mixture,
        timbre_mean, timbre_logvar,
        pitch_logits, pitch_annotation,
    )
    bce_loss = bce_loss_fn(pitch_logits, pitch_annotation)
    bt_loss = bt_loss_fn(eq, timbre_latent)
    
    # Check if any loss becomes NaN
    if torch.isnan(elbo_loss) or torch.isnan(bce_loss) or torch.isnan(bt_loss):
        print(f"NaN detected at step {step}: elbo_loss={elbo_loss.item()}, bce_loss={bce_loss.item()}, bt_loss={bt_loss.item()}")
        break  # Exit the training loop if NaN is encountered

    loss = elbo_loss + bce_loss + bt_loss
    print("ELBO:", elbo_loss.item(), "BCE:", bce_loss.item(), "BT:", bt_loss.item())
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    optimizer.step()
    
    # Log training metrics to wandb
    wandb.log({
        "train_loss": loss.item(),
        "elbo_loss": elbo_loss.item(),
        "bce_loss": bce_loss.item(),
        "bt_loss": bt_loss.item()
    })

    # Every few steps, evaluate on the validation set
    if step % 1000 == 0 and step > 100:  # Evaluate every 1000 steps
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            cnt = 0
            for batch in valid_data_loader:
                cnt += 1
                if cnt >= 100: break
                mixture = batch["mixture"].to(device)
                query = batch["query"].to(device)
                pitch_annotation = batch["pitch_annotation"].to(device) # pitch_label = batch["pitch_label"].to(device)
                    
                rec_mixture, pitch_latent, pitch_logits, timbre_latent, timbre_mean, timbre_logvar, eq = model(mixture, query)
                
                # Loss
                elbo_loss = elbo_loss_fn(
                    mixture, rec_mixture,
                    timbre_mean, timbre_logvar,
                    pitch_logits, pitch_annotation,
                )
                bce_loss = bce_loss_fn(pitch_logits, pitch_annotation)
                bt_loss = bt_loss_fn(eq, timbre_latent)
                
                loss = elbo_loss - bce_loss - bt_loss
                val_loss += loss.item()
        
        # Log validation loss to wandb
        wandb.log({"val_loss": val_loss})
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_steps = 0  # Reset the counter if validation improves
            
            # Optionally save the best model
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
        else:
            no_improvement_steps += 1000  # Increment the counter if no improvement
        
        # Early stopping condition
        if no_improvement_steps >= early_stop_patience:
            print(f"Early stopping triggered after {step} steps with best validation loss: {best_val_loss}")
            break
        

# Testing
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_data_loader:
        mixture = batch["mixture"].to(device)
        query = batch["query"].to(device)
        pitch_annotation = batch["pitch_annotation"].to(device) # pitch_label = batch["pitch_label"].to(device)
            
        rec_mixture, pitch_latent, pitch_logits, timbre_latent, timbre_mean, timbre_logvar, eq = model(mixture, query)
        
        # Loss
        elbo_loss = elbo_loss_fn(
            mixture, rec_mixture, # pitch_mean, pitch_logvar, 
            timbre_mean, timbre_logvar
        )
        bce_loss = bce_loss_fn(pitch_logits, pitch_annotation)
        bt_loss = bt_loss_fn(eq, timbre_latent)
        
        loss = elbo_loss - bce_loss - bt_loss
        test_loss += loss.item()

# Log test loss to wandb
wandb.log({"test_loss": test_loss})
wandb.finish()


