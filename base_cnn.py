import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "classification_dataset")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "base_cnn")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 60)
print("DEVICE:", device)
print("=" * 60)


# ============================================================
# 3. PARAMETERS
# ============================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# 4. DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

print("\nClasses:", class_names)
print("Number of classes:", NUM_CLASSES)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. BASE CNN MODEL
# ============================================================

class BaseCNN(nn.Module):

    def __init__(self, num_classes):
        super(BaseCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = BaseCNN(NUM_CLASSES).to(device)

print("\nModel:")
print(model)


# ============================================================
# 8. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. TRAINING
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

print("\n")
print("=" * 60)
print("STARTING BASE CNN TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_dataset)
    val_accuracy = val_correct / val_total


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                RESULTS_DIR,
                "base_cnn_best.pth"
            )
        )


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


# Restore best model
model.load_state_dict(best_model_weights)

print("\nTraining completed.")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    ),
    index=False
)


# ============================================================
# 11. VALIDATION LOSS & ACCURACY PLOTS
# ============================================================

epochs_range = range(1, EPOCHS + 1)


# Accuracy plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Base CNN - Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    ),
    dpi=300
)

plt.close()


# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Base CNN - Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 12. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, loader, dataset_name):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)


    # --------------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        y_true_one_hot = np.eye(
            NUM_CLASSES
        )[y_true]

        roc_auc = roc_auc_score(
            y_true_one_hot,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        roc_auc = float("nan")


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n")
    print("=" * 60)
    print(dataset_name.upper(), "RESULTS")
    print("=" * 60)

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # SENSITIVITY & SPECIFICITY
    # --------------------------------------------------------

    sensitivities = []
    specificities = []

    for i in range(NUM_CLASSES):

        TP = cm[i, i]

        FN = np.sum(cm[i, :]) - TP

        FP = np.sum(cm[:, i]) - TP

        TN = np.sum(cm) - (
            TP + FN + FP
        )

        sensitivity = (
            TP / (TP + FN)
            if (TP + FN) > 0
            else 0
        )

        specificity = (
            TN / (TN + FP)
            if (TN + FP) > 0
            else 0
        )

        sensitivities.append(
            sensitivity
        )

        specificities.append(
            specificity
        )


    sensitivity_avg = np.mean(
        sensitivities
    )

    specificity_avg = np.mean(
        specificities
    )

    print(
        f"Sensitivity : {sensitivity_avg:.4f}"
    )

    print(
        f"Specificity : {specificity_avg:.4f}"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)


    # Save classification report
    with open(
        os.path.join(
            RESULTS_DIR,
            f"{dataset_name.lower()}_classification_report.txt"
        ),
        "w"
    ) as f:

        f.write(report)


    # --------------------------------------------------------
    # CONFUSION MATRIX PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(
        f"Base CNN - {dataset_name} Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        range(NUM_CLASSES),
        class_names,
        rotation=45
    )

    plt.yticks(
        range(NUM_CLASSES),
        class_names
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{dataset_name.lower()}_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity_avg,
        "specificity": specificity_avg,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# 13. VALIDATION EVALUATION
# ============================================================

val_results = evaluate_model(
    model,
    val_loader,
    "Validation"
)


# ============================================================
# 14. TEST EVALUATION
# ============================================================

test_results = evaluate_model(
    model,
    test_loader,
    "Test"
)


# ============================================================
# 15. SAVE FINAL RESULTS
# ============================================================

results_table = pd.DataFrame([
    {
        "Model": "Base CNN",
        "Dataset": "Validation",
        **val_results
    },
    {
        "Model": "Base CNN",
        "Dataset": "Test",
        **test_results
    }
])

results_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "base_cnn_results.csv"
    ),
    index=False
)


# ============================================================
# 16. SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model": "Base CNN",
    "classes": class_names,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "device": str(device),
    "best_validation_accuracy": best_val_accuracy
}

with open(
    os.path.join(
        RESULTS_DIR,
        "model_info.json"
    ),
    "w"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )


print("\n")
print("=" * 60)
print("BASE CNN COMPLETE")
print("=" * 60)

print("\nResults saved in:")

print(RESULTS_DIR)

print("\nFiles generated:")

print("1. base_cnn_best.pth")
print("2. training_history.csv")
print("3. accuracy_curve.png")
print("4. loss_curve.png")
print("5. validation_confusion_matrix.png")
print("6. test_confusion_matrix.png")
print("7. validation_classification_report.txt")
print("8. test_classification_report.txt")
print("9. base_cnn_results.csv")
print("10. model_info.json")