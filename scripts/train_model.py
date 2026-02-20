import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import os
import time
import re # To extract epoch number from checkpoint filename

print("TensorFlow version:", tf.__version__)

# --- Verify GPU is available ---
# (GPU check code remains the same as previous script)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print(f"   Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}. If training fails, restart environment.")
else:
     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     print("!!! WARNING: No GPU detected by TensorFlow. Training WILL BE EXTREMELY SLOW. !!!")
     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# --- 1. Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8 # Keep batch size low for 4GB VRAM
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 25
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

TRAIN_DIR = 'processed_dataset/train'
VALIDATION_DIR = 'processed_dataset/validation'
AUTOTUNE = tf.data.AUTOTUNE

# --- Checkpoint Configuration ---
CHECKPOINT_DIR = "training_checkpoints" # Folder to save epoch checkpoints
CHECKPOINT_PATH_FORMAT = os.path.join(CHECKPOINT_DIR, "cp-{epoch:04d}.weights.h5") # File format for epoch checkpoints
BEST_MODEL_FILE = 'best_model_resumable.h5' # File for the best model overall

# --- Enable Mixed Precision ---
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f'Mixed precision policy set to: {policy.name}')

# --- 2. Data Pipeline ---
# (build_dataset, data_augmentation, preprocess functions remain the same as previous 'efficient' script)
def build_dataset(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory, labels='inferred', label_mode='binary', image_size=IMAGE_SIZE,
        interpolation='bicubic', batch_size=BATCH_SIZE, shuffle=True
    )
train_ds = build_dataset(TRAIN_DIR)
validation_ds = build_dataset(VALIDATION_DIR)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
], name="data_augmentation")

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    return tf.keras.applications.xception.preprocess_input(image), label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Build Model Function ---
def create_model():
    input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
    x = data_augmentation(input_tensor)
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False # Start frozen
    x = base_model.output
    x = tf.keras.layers.Activation('linear', dtype='float32')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid', dtype='float32')(x)
    model = Model(inputs=input_tensor, outputs=predictions, name="ResumableDeepfakeDetector")
    return model, base_model

model, base_model = create_model()

# --- 4. Define Optimizers (Needed Before Loading Checkpoint with Optimizer State) ---
initial_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
initial_optimizer = mixed_precision.LossScaleOptimizer(initial_optimizer) # Wrap for mixed precision

fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
fine_tune_optimizer = mixed_precision.LossScaleOptimizer(fine_tune_optimizer) # Wrap for mixed precision

# --- 5. Load Latest Checkpoint if Exists ---
initial_epoch = 0
latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)

if latest_checkpoint:
    print(f"\n--- Resuming training from checkpoint: {latest_checkpoint} ---")
    # Extract epoch number from filename (assuming format cp-XXXX.weights.h5)
    try:
        match = re.search(r"cp-(\d{4})\.weights\.h5", latest_checkpoint)
        if match:
            initial_epoch = int(match.group(1))
            print(f"   Resuming from the end of epoch {initial_epoch}.")
            initial_epoch += 1 # Start training *after* the last saved epoch
        else:
             print("   Could not parse epoch number from checkpoint filename, starting epoch count from 0.")
             initial_epoch = 0
    except Exception as e:
        print(f"   Error parsing epoch number: {e}. Starting epoch count from 0.")
        initial_epoch = 0

    # Compile the model *before* loading weights to restore optimizer state
    # We need to determine which phase we are in based on the initial_epoch
    if initial_epoch < INITIAL_EPOCHS:
         print("   Checkpoint is within Initial Training phase.")
         model.compile(optimizer=initial_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
         base_model.trainable = False # Ensure base is frozen if resuming in this phase
    else:
         print("   Checkpoint is within Fine-Tuning phase.")
         base_model.trainable = True # Unfreeze for fine-tuning
         fine_tune_at = 105
         for layer in base_model.layers[:fine_tune_at]:
             layer.trainable = False
         model.compile(optimizer=fine_tune_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
         print("   Model configured for fine-tuning.")

    # Load the weights (restores model and optimizer state if compiled first)
    model.load_weights(latest_checkpoint)
    print("   Model weights (and optimizer state) loaded successfully.")

else:
    print("\n--- No checkpoint found. Starting training from scratch. ---")
    # Compile for initial training if starting fresh
    model.compile(optimizer=initial_optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# --- 6. Define Callbacks ---
# Callback to save weights every epoch (for resuming)
epoch_checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH_FORMAT,
    save_weights_only=True, # Saves model + optimizer state
    save_freq='epoch', # Save after every epoch
    verbose=1)

# Callback to save only the best model overall (based on validation accuracy)
best_model_callback = ModelCheckpoint(
    BEST_MODEL_FILE, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
)

# Combine callbacks
callbacks = [epoch_checkpoint_callback, best_model_callback, early_stopping, reduce_lr]


# --- 7. Determine Training Phase and Run ---
print(f"\n--- Starting/Resuming Training from Epoch {initial_epoch} ---")
model.summary() # Print summary after potential weight loading / recompiling

start_time = time.time()

# Run model.fit ONCE, letting it handle both phases based on initial_epoch
history = model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS, # Train up to the total number of epochs
    initial_epoch=initial_epoch, # Start from the determined epoch
    validation_data=validation_ds,
    callbacks=callbacks
)

end_time = time.time()
print(f"--- Training finished in {(end_time - start_time)/60:.2f} minutes ---")

# --- Optional: Logic to switch to fine-tuning if initial training finishes ---
# NOTE: The provided structure simplifies this by letting fit() run to TOTAL_EPOCHS.
# If you strictly want separate phases even when resuming, the logic gets more complex.
# The current approach is simpler and relies on callbacks like ReduceLROnPlateau
# and EarlyStopping to manage the learning effectively across the entire run.

# --- If you manually stopped *before* fine-tuning started and want to ensure it runs: ---
# This part is slightly redundant if initial_epoch already covers the transition,
# but can be added as a safeguard if needed.
final_epoch_reached = history.epoch[-1] + 1 # Epochs are 0-indexed
if final_epoch_reached >= INITIAL_EPOCHS and base_model.trainable == False:
    print("\n--- Initial training phase complete. Manually triggering fine-tuning setup ---")
    print("   (This might happen if training stopped exactly at INITIAL_EPOCHS)")

    # Load best weights achieved so far
    print("   Loading best weights achieved...")
    if os.path.exists(BEST_MODEL_FILE):
        model.load_weights(BEST_MODEL_FILE)
    else:
        print("   Warning: Best model file not found, continuing with current weights.")

    # Apply fine-tuning setup
    base_model.trainable = True
    fine_tune_at = 105
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print("   Re-compiling model for fine-tuning...")
    model.compile(optimizer=fine_tune_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    print(f"\n--- Continuing Training for Fine-Tuning (Up to {TOTAL_EPOCHS} total epochs) ---")
    start_time_ft = time.time()
    history_fine_tune = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=final_epoch_reached, # Continue from the next epoch
        validation_data=validation_ds,
        callbacks=callbacks # Continue using callbacks
    )
    end_time_ft = time.time()
    print(f"--- Fine-Tuning finished in {(end_time_ft - start_time_ft)/60:.2f} minutes ---")


print(f"\n\nTraining complete! Check '{CHECKPOINT_DIR}' for epoch checkpoints.")
print(f"The overall best model based on validation accuracy was saved as '{BEST_MODEL_FILE}'")