from pathlib import Path
import copy
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "classification_dataset"
RESULTS_DIR = BASE_DIR / "results" / "transfer_learning"

# Create output directory automatically
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SELECT MODEL
# ============================================================

if len(sys.argv) < 2:
    print("\nPlease specify a model:")
    print("python3 transfer_learning.py resnet50")
    print("python3 transfer_learning.py mobilenetv2")
    print("python3 transfer_learning.py efficientnetb0")
    sys.exit(1)

MODEL_NAME = sys.argv[1].lower()


MODEL_NAMES = {
    "resnet50": "ResNet50",
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0"
}

if MODEL_NAME not in MODEL_NAMES:
    print("Invalid model name.")
    print("Use: resnet50, mobilenetv2, or efficientnetb0")
    sys.exit(1)

DISPLAY_NAME = MODEL_NAMES[MODEL_NAME]

MODEL_DIR = RESULTS_DIR / DISPLAY_NAME

# THIS PREVENTS THE PREVIOUS ERROR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print(f"TRAINING MODEL: {DISPLAY_NAME}")
print("=" * 60)

print(f"Dataset: {DATA_DIR}")
print(f"Results: {MODEL_DIR}")


# ============================================================
# 3. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nUsing Apple GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nUsing NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("\nUsing CPU")


# ============================================================
# 4. SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
PATIENCE = 2

NUM_WORKERS = 0


# ============================================================
# 5. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. FIND DATASET FOLDERS
# ============================================================

train_dir = DATA_DIR / "train"

if (DATA_DIR / "validation").exists():
    val_dir = DATA_DIR / "validation"
elif (DATA_DIR / "val").exists():
    val_dir = DATA_DIR / "val"
else:
    raise FileNotFoundError(
        "Could not find validation folder. "
        "Expected 'validation' or 'val'."
    )

test_dir = DATA_DIR / "test"


# ============================================================
# 7. LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=eval_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("\nClasses:")
print(class_names)

print(f"\nTraining images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images:    {len(test_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 8. CREATE MODEL
# ============================================================

print(f"\nLoading pretrained {DISPLAY_NAME}...")


if MODEL_NAME == "resnet50":

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )


elif MODEL_NAME == "mobilenetv2":

    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )


elif MODEL_NAME == "efficientnetb0":

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )


model = model.to(device)


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

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

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================

def validate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# 12. TRAIN
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

patience_counter = 0

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_acc = validate(
        model,
        val_loader
    )

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_model_weights,
            MODEL_DIR / f"{DISPLAY_NAME}_best.pth"
        )

        patience_counter = 0

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break


# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_model_weights)

print("\nBest model loaded.")


# ============================================================
# 14. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)


# ============================================================
# 15. ACCURACY CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"{DISPLAY_NAME} Accuracy Curve")
plt.legend()
plt.grid(True)

plt.savefig(
    MODEL_DIR / "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{DISPLAY_NAME} Loss Curve")
plt.legend()
plt.grid(True)

plt.savefig(
    MODEL_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

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

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# 18. VALIDATION EVALUATION
# ============================================================

val_labels, val_predictions, val_probabilities = evaluate(
    model,
    val_loader
)

val_accuracy = accuracy_score(
    val_labels,
    val_predictions
)

val_precision = precision_score(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0
)

val_recall = recall_score(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0
)

val_f1 = f1_score(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0
)

try:
    val_auc = roc_auc_score(
        label_binarize(val_labels, classes=range(num_classes)),
        val_probabilities,
        multi_class="ovr",
        average="macro"
    )
except:
    val_auc = 0.0


# ============================================================
# 19. TEST EVALUATION
# ============================================================

test_labels, test_predictions, test_probabilities = evaluate(
    model,
    test_loader
)

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

try:
    test_auc = roc_auc_score(
        label_binarize(
            test_labels,
            classes=range(num_classes)
        ),
        test_probabilities,
        multi_class="ovr",
        average="macro"
    )
except:
    test_auc = 0.0


# ============================================================
# 20. SPECIFICITY
# ============================================================

def calculate_specificity(labels, predictions):

    cm = confusion_matrix(
        labels,
        predictions,
        labels=range(num_classes)
    )

    specificities = []

    for i in range(num_classes):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = cm.sum() - (tp + fn + fp)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        specificities.append(specificity)

    return np.mean(specificities)


val_specificity = calculate_specificity(
    val_labels,
    val_predictions
)

test_specificity = calculate_specificity(
    test_labels,
    test_predictions
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(labels, predictions, filename, title):

    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(title)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(num_classes),
        class_names,
        rotation=45
    )

    plt.yticks(
        range(num_classes),
        class_names
    )

    for i in range(num_classes):
        for j in range(num_classes):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_confusion_matrix(
    val_labels,
    val_predictions,
    "validation_confusion_matrix.png",
    f"{DISPLAY_NAME} Validation Confusion Matrix"
)

save_confusion_matrix(
    test_labels,
    test_predictions,
    "test_confusion_matrix.png",
    f"{DISPLAY_NAME} Test Confusion Matrix"
)


# ============================================================
# 22. CLASSIFICATION REPORTS
# ============================================================

val_report = classification_report(
    val_labels,
    val_predictions,
    target_names=class_names,
    zero_division=0
)

test_report = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    zero_division=0
)

with open(
    MODEL_DIR / "validation_classification_report.txt",
    "w"
) as f:

    f.write(val_report)


with open(
    MODEL_DIR / "test_classification_report.txt",
    "w"
) as f:

    f.write(test_report)


# ============================================================
# 23. ROC CURVE
# ============================================================

binary_test_labels = label_binarize(
    test_labels,
    classes=range(num_classes)
)

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(class_names):

    fpr, tpr, _ = roc_curve(
        binary_test_labels[:, i],
        test_probabilities[:, i]
    )

    plt.plot(
        fpr,
        tpr,
        label=class_name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"{DISPLAY_NAME} ROC Curve"
)

plt.legend()

plt.grid(True)

plt.savefig(
    MODEL_DIR / "test_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. RESULTS
# ============================================================

results = {
    "Model": DISPLAY_NAME,

    "Validation Accuracy": val_accuracy,
    "Validation Precision": val_precision,
    "Validation Sensitivity": val_recall,
    "Validation Specificity": val_specificity,
    "Validation F1": val_f1,
    "Validation ROC-AUC": val_auc,

    "Test Accuracy": test_accuracy,
    "Test Precision": test_precision,
    "Test Sensitivity": test_recall,
    "Test Specificity": test_specificity,
    "Test F1": test_f1,
    "Test ROC-AUC": test_auc
}


results_df = pd.DataFrame([results])

results_df.to_csv(
    MODEL_DIR / "model_results.csv",
    index=False
)


# ============================================================
# 25. MODEL INFO
# ============================================================

model_info = {
    "model": DISPLAY_NAME,
    "num_classes": num_classes,
    "classes": class_names,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "device": str(device),
    "pretrained": True,
    "backbone_frozen": True
}

with open(
    MODEL_DIR / "model_info.json",
    "w"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print(f"{DISPLAY_NAME} TRAINING COMPLETED")
print("=" * 60)

print(f"\nTest Accuracy:      {test_accuracy:.4f}")
print(f"Test Precision:     {test_precision:.4f}")
print(f"Test Sensitivity:   {test_recall:.4f}")
print(f"Test Specificity:   {test_specificity:.4f}")
print(f"Test F1 Score:      {test_f1:.4f}")
print(f"Test ROC-AUC:       {test_auc:.4f}")

print(f"\nAll files saved in:")
print(MODEL_DIR)

print("\nDone!")