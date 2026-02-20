import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import re
# Import amp (Automatic Mixed Precision)
from torch.cuda.amp import GradScaler, autocast

# --- Function Definitions (Keep build_dataset and create_model) ---
def build_dataset(directory, image_size, batch_size, phase):
    # (Same as before)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print(f"Building dataset for phase: {phase} from {directory}")
    image_dataset = datasets.ImageFolder(directory, data_transforms[phase])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=(phase == 'train'),
                                             num_workers=min(os.cpu_count(), 4),
                                             pin_memory=True)
    return dataloader, len(image_dataset), image_dataset.classes

def create_model(model_name, num_classes, device):
    # (Same as before)
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Model {model_name} not configured")

    # DON'T freeze here initially, as we'll load weights into the full structure
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print(f"Model {model_name} structure created.")
    return model

# --- Main Execution Block ---
if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # --- 1. Configuration ---
    TRAIN_DIR = 'processed_dataset/train'
    VALIDATION_DIR = 'processed_dataset/validation'
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 1
    IMAGE_SIZE = 224
    BATCH_SIZE = 16 # Keep consistent with previous run or adjust if needed
    # --- Fine-Tuning Specific Config ---
    FINE_TUNE_EPOCHS = 25 # Number of epochs for this fine-tuning run
    FINE_TUNE_LR = 1e-5 # Low learning rate for fine-tuning
    # --- Files ---
    LOAD_WEIGHTS_FILE = 'best_model_pytorch_accuracy60.pth' # Load the best model from previous run
    BEST_MODEL_FILE = 'best_model_pytorch_finetuned.pth' # Save the best fine-tuned model here
    # Checkpoints for fine-tuning phase (optional but good practice)
    CHECKPOINT_DIR_FT = os.path.join('.', "training_checkpoints_pytorch_ft")
    CHECKPOINT_FILE_FT = os.path.join(CHECKPOINT_DIR_FT, 'latest_checkpoint_ft.pth')
    os.makedirs(CHECKPOINT_DIR_FT, exist_ok=True)


    # --- 2. Setup GPU Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else: print("⚠️ WARNING: CUDA not available, using CPU.")

    # --- 3. Data Loading ---
    print("Building datasets...")
    dataloaders = {}
    dataset_sizes = {}
    class_names = None
    dataloaders['train'], dataset_sizes['train'], class_names = build_dataset(TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE, 'train')
    dataloaders['validation'], dataset_sizes['validation'], _ = build_dataset(VALIDATION_DIR, IMAGE_SIZE, BATCH_SIZE, 'validation')
    if class_names: print(f"Classes found: {class_names}")
    print(f"Training images: {dataset_sizes.get('train', 0)}, Validation images: {dataset_sizes.get('validation', 0)}")

    # --- 4. Load Model and Prepare for Fine-Tuning ---
    model = create_model(MODEL_NAME, NUM_CLASSES, device)

    print(f"\n--- Loading best weights from: {LOAD_WEIGHTS_FILE} ---")
    if os.path.exists(LOAD_WEIGHTS_FILE):
        model.load_state_dict(torch.load(LOAD_WEIGHTS_FILE, map_location=device, weights_only=True))
        print("   Best weights loaded successfully.")
    else:
        print(f"   ERROR: Best weights file not found at {LOAD_WEIGHTS_FILE}. Cannot proceed.")
        exit()

    print("\n--- Configuring model for Fine-Tuning ---")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    # Optional: Re-freeze early layers (e.g., first few blocks of ResNet)
    # fine_tune_freeze_until = 'layer3' # Example
    # for name, child in model.named_children():
    #      if name == fine_tune_freeze_until:
    #          break
    #      for param in child.parameters():
    #          param.requires_grad = False
    #      print(f"   Froze layer: {name}")
    print("   All layers unfrozen (adjust freezing logic if desired).")


    # --- 5. Define Loss, Optimizer, Scheduler for Fine-Tuning ---
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer for fine-tuning - uses all parameters now
    current_optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(current_optimizer, mode='min', factor=0.2, patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # --- 6. Load Fine-Tuning Checkpoint (If Exists) ---
    start_epoch = 0
    best_acc = 0.0 # Reset best accuracy for this fine-tuning phase
    if os.path.isfile(CHECKPOINT_FILE_FT):
        print(f"--- Loading fine-tuning checkpoint '{CHECKPOINT_FILE_FT}' ---")
        checkpoint = torch.load(CHECKPOINT_FILE_FT, map_location=device, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0) # Load previous best FT accuracy
        # Load model weights from FT checkpoint (overwrites the initial best weights)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            current_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   Optimizer state loaded.")
        except Exception as e:
            print(f"   Warning: Could not load optimizer state: {e}. Re-initializing.")
            current_optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR) # Re-init if needed
        if 'scheduler_state_dict' in checkpoint:
            try: scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("   Scheduler state loaded.")
            except: print("   Warning: Could not load scheduler state."); scheduler = lr_scheduler.ReduceLROnPlateau(current_optimizer, mode='min', factor=0.2, patience=3, verbose=True)
        if 'scaler_state_dict' in checkpoint and torch.cuda.is_available():
            try: scaler.load_state_dict(checkpoint['scaler_state_dict']); print("   Gradient Scaler state loaded.")
            except: print("   Warning: Could not load GradScaler state."); scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        print(f"   Resuming fine-tuning from epoch {start_epoch}, Best FT accuracy so far: {best_acc:.4f}")
    else:
        print(f"--- No fine-tuning checkpoint found. Starting fine-tuning from epoch 0. ---")


    # --- 7. Fine-Tuning Loop ---
    print(f"\n--- Starting/Resuming Fine-Tuning from Epoch {start_epoch} up to {FINE_TUNE_EPOCHS} ---")
    since = time.time()
    if os.path.exists(BEST_MODEL_FILE) and best_acc == 0.0 and start_epoch == 0:
         print(f"Loading previous best fine-tuned model weights from {BEST_MODEL_FILE}")
         model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True))
    best_model_wts = copy.deepcopy(model.state_dict())

    patience_counter = 0
    early_stopping_patience = 10 # Patience for fine-tuning

    # Run only for the number of fine-tuning epochs
    for epoch in range(start_epoch, FINE_TUNE_EPOCHS):
        print(f'Fine-Tune Epoch {epoch+1}/{FINE_TUNE_EPOCHS}')
        print('-' * 10)

        # --- Training and Validation Phases ---
        for phase in ['train', 'validation']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss = 0.0
            running_corrects = 0
            data_iterator = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} FT Epoch {epoch+1}")

            for inputs, labels in data_iterator:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                current_optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(current_optimizer)
                    scaler.update()

                preds_prob = torch.sigmoid(outputs.float())
                preds = (preds_prob > 0.5).float()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # TQDM update logic (corrected)
                samples_processed = (data_iterator.n + 1) * inputs.size(0) if data_iterator.n is not None else inputs.size(0)
                epoch_loss_running = running_loss / samples_processed if samples_processed > 0 else 0
                epoch_acc_running = running_corrects.double() / samples_processed if samples_processed > 0 else 0
                data_iterator.set_postfix(loss=f"{epoch_loss_running:.4f}", acc=f"{epoch_acc_running:.4f}")

            epoch_loss = running_loss / dataset_sizes[phase] if dataset_sizes[phase] > 0 else 0
            epoch_acc = running_corrects.double() / dataset_sizes[phase] if dataset_sizes[phase] > 0 else 0
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # --- Saving Best FT Model ---
            if phase == 'validation':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), BEST_MODEL_FILE) # Save best FT weights
                    print(f'✅ New best fine-tune validation accuracy: {best_acc:.4f}, saving best model to {BEST_MODEL_FILE}')
                    patience_counter = 0
                else:
                    patience_counter += 1

        # --- Save FT Checkpoint ---
        print(f"   Saving fine-tune checkpoint for epoch {epoch}...")
        checkpoint_data = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': current_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_acc': best_acc, 'loss': epoch_loss,
        }
        if torch.cuda.is_available(): checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint_data, CHECKPOINT_FILE_FT) # Save FT checkpoint
        print(f"   Fine-tune checkpoint saved to {CHECKPOINT_FILE_FT}")

        # --- Early Stopping Check ---
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping triggered during fine-tuning after {patience_counter} epochs with no improvement.")
            break
        print()

    # --- End of Fine-Tuning ---
    time_elapsed = time.time() - since
    print(f'\nFine-tuning complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best fine-tune validation Acc saved: {best_acc:4f}')

    # Load best FT model weights
    print(f"Loading best fine-tuned model weights from {BEST_MODEL_FILE}...")
    if os.path.exists(BEST_MODEL_FILE):
        model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True))
        print("Best fine-tuned model weights loaded.")
    else:
        print("Warning: Best fine-tuned model file not found.")

    print(f"\n\nFine-tuning finished! Best model weights saved to '{BEST_MODEL_FILE}'.")
    print(f"Latest fine-tuning state for resuming in '{CHECKPOINT_FILE_FT}'.")