"""
Pneumonia Detection from Paediatric Chest X-Ray Images
using a Convolutional Neural Network (TensorFlow / Keras)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'x-ray_image')

TRAIN_PATH = os.path.join(DATA_DIR, 'train')
TEST_PATH = os.path.join(DATA_DIR, 'test')
VAL_PATH = os.path.join(DATA_DIR, 'val')

SAVE_DIR = os.path.join(ROOT_DIR, 'outputs')
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
RANDOM_STATE = 42


# -------------------------------------------------------
# 1. Dataset Overview
# -------------------------------------------------------
def print_dataset_info():
    """Show how many images exist in each split and class."""
    print("\n" + "=" * 58)
    print("  DATASET OVERVIEW")
    print("=" * 58)

    folders = {'Training': TRAIN_PATH, 'Testing': TEST_PATH, 'Validation': VAL_PATH}
    total = 0

    for name, path in folders.items():
        normal = len(os.listdir(os.path.join(path, 'NORMAL')))
        pneumonia = len(os.listdir(os.path.join(path, 'PNEUMONIA')))
        subtotal = normal + pneumonia
        total += subtotal

        print(f"\n  {name}:")
        print(f"    NORMAL    -> {normal}")
        print(f"    PNEUMONIA -> {pneumonia}")
        print(f"    Subtotal  -> {subtotal}")
        print(f"    Ratio (P/N): {pneumonia / normal:.2f}")

    print(f"\n  Total images: {total}")
    print("=" * 58)


def visualise_samples():
    """Plot a row of images from each class."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle('Example Chest X-Ray Images', fontsize=16, fontweight='bold')

    for row, category in enumerate(['NORMAL', 'PNEUMONIA']):
        src = os.path.join(TRAIN_PATH, category)
        filenames = sorted(os.listdir(src))[:5]
        for col, fname in enumerate(filenames):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(src, fname), target_size=IMAGE_SIZE
            )
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(category, fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'sample_xrays.png'), dpi=150, bbox_inches='tight')
    plt.show()


# -------------------------------------------------------
# 2. Data Loading and Augmentation
# -------------------------------------------------------
def prepare_generators():
    """
    Build ImageDataGenerators for train / val / test.
    Training images are augmented; val and test are only rescaled.
    """
    augmentor = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    rescaler = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = augmentor.flow_from_directory(
        TRAIN_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_STATE,
        shuffle=True
    )
    val_gen = rescaler.flow_from_directory(
        VAL_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_STATE,
        shuffle=False
    )
    test_gen = rescaler.flow_from_directory(
        TEST_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_STATE,
        shuffle=False
    )

    print(f"\n  Classes: {train_gen.class_indices}")
    print(f"  Train samples: {train_gen.samples}")
    print(f"  Val samples:   {val_gen.samples}")
    print(f"  Test samples:  {test_gen.samples}\n")

    return train_gen, val_gen, test_gen


# -------------------------------------------------------
# 3. CNN Architecture
# -------------------------------------------------------
def build_model():
    """
    Construct a Sequential CNN with four conv blocks.
    Filter sizes increase from 32 to 256. BatchNormalization
    and Dropout are used for regularisation throughout.
    """
    cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(*IMAGE_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    cnn.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    cnn.summary()
    return cnn


# -------------------------------------------------------
# 4. Class Weights
# -------------------------------------------------------
def calculate_class_weights(generator):
    """Return weights that compensate for class imbalance."""
    counts = np.bincount(generator.classes)
    n = generator.samples
    weights = {0: n / (2 * counts[0]), 1: n / (2 * counts[1])}
    print(f"  Class weights — NORMAL: {weights[0]:.4f}, PNEUMONIA: {weights[1]:.4f}")
    return weights


# -------------------------------------------------------
# 5. Training Loop
# -------------------------------------------------------
def train(model, train_gen, val_gen):
    """Fit the model with early stopping and LR scheduling."""
    weights = calculate_class_weights(train_gen)

    cb = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(SAVE_DIR, 'best_weights.keras'),
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("\n" + "=" * 58)
    print("  TRAINING STARTED")
    print("=" * 58 + "\n")

    history = model.fit(
        train_gen,
        epochs=NUM_EPOCHS,
        validation_data=val_gen,
        class_weight=weights,
        callbacks=cb,
        verbose=1
    )
    return history


# -------------------------------------------------------
# 6. Plotting
# -------------------------------------------------------
def plot_history(history):
    """Accuracy and loss curves."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    ax_acc.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax_acc.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax_acc.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')
    ax_acc.grid(True, alpha=0.3)

    ax_loss.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax_loss.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax_loss.set_title('Loss', fontsize=14, fontweight='bold')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'accuracy_loss_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.show()


def plot_cm(y_true, y_pred, labels):
    """Draw a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150,
                bbox_inches='tight')
    plt.show()


# -------------------------------------------------------
# 7. Evaluation
# -------------------------------------------------------
def evaluate(model, test_gen):
    """Run the model on the test set and report metrics."""
    print("\n" + "=" * 58)
    print("  EVALUATION ON TEST SET")
    print("=" * 58)

    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\n  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    preds = model.predict(test_gen, verbose=1)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=labels)
    print("\n" + report)

    with open(os.path.join(SAVE_DIR, 'test_results.txt'), 'w') as f:
        f.write("Pneumonia Detection — Test Results\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Loss:     {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n\n")
        f.write(report)

    plot_cm(y_true, y_pred, labels)
    return test_loss, test_acc


# -------------------------------------------------------
# 8. Single-Image Prediction
# -------------------------------------------------------
def predict_image(model, filepath):
    """Load one image and print the predicted label."""
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=IMAGE_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = model.predict(arr, verbose=0)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f"{label}  ({confidence:.1%})", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"  -> {label} ({confidence:.1%})")
    return label, confidence


# -------------------------------------------------------
# 9. Main
# -------------------------------------------------------
def main():
    print("\n" + "=" * 58)
    print("  PNEUMONIA DETECTION — CNN PIPELINE")
    print("  Built with TensorFlow & Keras")
    print("=" * 58)

    print_dataset_info()
    visualise_samples()

    train_gen, val_gen, test_gen = prepare_generators()

    model = build_model()

    history = train(model, train_gen, val_gen)
    plot_history(history)

    test_loss, test_acc = evaluate(model, test_gen)

    model.save(os.path.join(SAVE_DIR, 'pneumonia_cnn.keras'))
    print(f"\n  Model saved -> outputs/pneumonia_cnn.keras")

    normal_sample = os.path.join(
        TEST_PATH, 'NORMAL', sorted(os.listdir(os.path.join(TEST_PATH, 'NORMAL')))[0]
    )
    pneumonia_sample = os.path.join(
        TEST_PATH, 'PNEUMONIA', sorted(os.listdir(os.path.join(TEST_PATH, 'PNEUMONIA')))[0]
    )

    print("\n  --- Sample Predictions ---")
    print("  Normal X-ray:")
    predict_image(model, normal_sample)
    print("  Pneumonia X-ray:")
    predict_image(model, pneumonia_sample)

    print("\n" + "=" * 58)
    print(f"  COMPLETE — Final test accuracy: {test_acc:.2%}")
    print("=" * 58 + "\n")


if __name__ == '__main__':
    main()
