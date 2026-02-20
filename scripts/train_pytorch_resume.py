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
import re # To potentially parse epoch from filename if needed
# Import amp (Automatic Mixed Precision)
from torch.cuda.amp import GradScaler, autocast

# --- Function Definitions ---

def build_dataset(directory, image_size, batch_size, phase):
    """Builds a data pipeline with robust augmentations for training."""
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
    """Loads a pre-trained model and adapts its final layer."""
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Model {model_name} not configured")

    for param in model.parameters(): # Start frozen
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print(f"Model {model_name} loaded and classifier adapted.")
    return model

# --- Main Execution Block ---
if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # --- 1. Configuration (Prioritizing Accuracy, ~60 Epochs) ---
    TRAIN_DIR = 'processed_dataset/train'
    VALIDATION_DIR = 'processed_dataset/validation'
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 1
    IMAGE_SIZE = 224
    BATCH_SIZE = 16 # Adjust based on GPU VRAM
    TOTAL_EPOCHS = 60
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-5
    FINE_TUNE_START_EPOCH = 25

    # --- Checkpoint Configuration ---
    DRIVE_SAVE_DIR = '.'
    CHECKPOINT_DIR = os.path.join(DRIVE_SAVE_DIR, "training_checkpoints_pytorch_acc60")
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint_acc60.pth')
    BEST_MODEL_FILE = os.path.join(DRIVE_SAVE_DIR, 'best_model_pytorch_accuracy60.pth')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 2. Setup GPU Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else: print("⚠️ WARNING: CUDA not available, using CPU.")

    # --- 3. Data Augmentation and Loading ---
    print("Building datasets...")
    dataloaders = {}
    dataset_sizes = {}
    class_names = None
    dataloaders['train'], dataset_sizes['train'], class_names = build_dataset(TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE, 'train')
    dataloaders['validation'], dataset_sizes['validation'], _ = build_dataset(VALIDATION_DIR, IMAGE_SIZE, BATCH_SIZE, 'validation')
    if class_names: print(f"Classes found: {class_names}")
    print(f"Training images: {dataset_sizes.get('train', 0)}, Validation images: {dataset_sizes.get('validation', 0)}")

    # --- 4. Load Pre-trained Model ---
    model = create_model(MODEL_NAME, NUM_CLASSES, device)

    # --- 5. Define Loss, Optimizers, Scheduler ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer_initial = optim.Adam(model.fc.parameters(), lr=INITIAL_LR)
    optimizer_finetune = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_initial, mode='min', factor=0.2, patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # --- 6. Load Checkpoint (If Exists) ---
    start_epoch = 0
    best_acc = 0.0
    current_optimizer = optimizer_initial
    # (Checkpoint loading logic - kept the same)
    if os.path.isfile(CHECKPOINT_FILE):
        print(f"--- Loading checkpoint '{CHECKPOINT_FILE}' ---")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])

        if start_epoch < FINE_TUNE_START_EPOCH:
            print(f"   Resuming in Initial Training phase (Epoch {start_epoch}).")
            try:
                optimizer_initial.load_state_dict(checkpoint['optimizer_state_dict'])
                current_optimizer = optimizer_initial
                for param in model.parameters(): param.requires_grad = False
                for param in model.fc.parameters(): param.requires_grad = True
            except Exception as e: print(f"   Warning: Could not load initial optimizer state: {e}. Re-initializing."); current_optimizer = optim.Adam(model.fc.parameters(), lr=INITIAL_LR)
        else:
            print(f"   Resuming in Fine-Tuning phase (Epoch {start_epoch}).")
            for param in model.parameters(): param.requires_grad = True
            try:
                optimizer_finetune.load_state_dict(checkpoint['optimizer_state_dict'])
                current_optimizer = optimizer_finetune
            except Exception as e: print(f"   Warning: Could not load fine-tune optimizer state: {e}. Re-initializing."); current_optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
            print("   Model configured for fine-tuning.")

        scheduler.optimizer = current_optimizer
        if 'scheduler_state_dict' in checkpoint:
            try: scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("   Scheduler state loaded.")
            except: print("   Warning: Could not load scheduler state."); scheduler = lr_scheduler.ReduceLROnPlateau(current_optimizer, mode='min', factor=0.2, patience=3, verbose=True)
        if 'scaler_state_dict' in checkpoint and torch.cuda.is_available():
            try: scaler.load_state_dict(checkpoint['scaler_state_dict']); print("   Gradient Scaler state loaded.")
            except: print("   Warning: Could not load GradScaler state."); scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        print(f"   Resuming from epoch {start_epoch}, Best validation accuracy so far: {best_acc:.4f}")
    else:
        print(f"--- No checkpoint found at '{CHECKPOINT_FILE}'. Starting training from scratch. ---")
        current_optimizer = optimizer_initial


    # --- 7. Define Callbacks (Conceptual) ---
    patience_counter = 0
    early_stopping_patience = 10

    # --- 8. Training Loop ---
    print(f"\n--- Starting/Resuming Training from Epoch {start_epoch} up to {TOTAL_EPOCHS} ---")
    since = time.time()
    if os.path.exists(BEST_MODEL_FILE) and best_acc == 0.0 and start_epoch == 0:
         print(f"Loading previous best model weights from {BEST_MODEL_FILE}")
         model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True))
    best_model_wts = copy.deepcopy(model.state_dict())

    # --- Main Training Loop ---
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f'Epoch {epoch+1}/{TOTAL_EPOCHS}')
        print('-' * 10)

        # --- Switch to Fine-Tuning Phase Logic ---
        if epoch == FINE_TUNE_START_EPOCH:
            print("\n--- Switching to Fine-Tuning Phase ---")
            for param in model.parameters(): param.requires_grad = True
            current_optimizer = optimizer_finetune
            scheduler.optimizer = current_optimizer
            print(f"   Optimizer switched to Adam with LR={FINE_TUNE_LR}")
            print("   All base layers unfrozen for fine-tuning.")

        # --- Training and Validation Phases ---
        for phase in ['train', 'validation']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            data_iterator = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
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

                # --- CORRECTED TQDM UPDATE ---
                # Calculate samples processed *so far* in this epoch
                samples_processed = (data_iterator.n + 1) * inputs.size(0) if data_iterator.n is not None else inputs.size(0)
                # Calculate running average metrics for display
                epoch_loss_running = running_loss / samples_processed if samples_processed > 0 else 0
                epoch_acc_running = running_corrects.double() / samples_processed if samples_processed > 0 else 0
                # Update the progress bar description
                data_iterator.set_postfix(loss=f"{epoch_loss_running:.4f}", acc=f"{epoch_acc_running:.4f}")
                # --- END OF CORRECTION ---

            # Calculate final epoch metrics
            epoch_loss = running_loss / dataset_sizes[phase] if dataset_sizes[phase] > 0 else 0
            epoch_acc = running_corrects.double() / dataset_sizes[phase] if dataset_sizes[phase] > 0 else 0

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # --- Saving Best Model ---
            if phase == 'validation':
                scheduler.step(epoch_loss) # Step scheduler based on validation loss

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), BEST_MODEL_FILE)
                    print(f'✅ New best validation accuracy: {best_acc:.4f}, saving best model to {BEST_MODEL_FILE}')
                    patience_counter = 0 # Reset patience
                else:
                    patience_counter += 1

        # --- Save Checkpoint ---
        print(f"   Saving checkpoint for epoch {epoch}...")
        checkpoint_data = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': current_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_acc': best_acc, 'loss': epoch_loss,
        }
        if torch.cuda.is_available(): checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint_data, CHECKPOINT_FILE)
        print(f"   Checkpoint saved to {CHECKPOINT_FILE}")

        # --- Early Stopping Check ---
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs with no improvement.")
            break # Exit the main training loop
        print()

    # --- End of Training ---
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc saved: {best_acc:4f}')

    # Load best model weights before finishing
    print(f"Loading best model weights from {BEST_MODEL_FILE}...")
    if os.path.exists(BEST_MODEL_FILE):
        model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True)) # Use weights_only=True
        print("Best model weights loaded.")
    else:
        print("Warning: Best model file not found.")

    print(f"\n\nTraining finished! Best model weights saved to '{BEST_MODEL_FILE}'.")
    print(f"Latest training state for resuming in '{CHECKPOINT_FILE}'.")