from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
TL_DIR = RESULTS_DIR / "transfer_learning"

OUTPUT_DIR = RESULTS_DIR / "final_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FINAL MODEL RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "Base CNN",
        "Accuracy": 0.7304,
        "Precision": 0.7306,
        "Sensitivity": 0.6658,
        "Specificity": 0.9083,
        "F1 Score": 0.6929,
        "ROC-AUC": 0.9271
    },
    {
        "Model": "ResNet50",
        "Accuracy": 0.8256,
        "Precision": 0.8023,
        "Sensitivity": 0.8078,
        "Specificity": 0.9435,
        "F1 Score": 0.8039,
        "ROC-AUC": 0.9620
    },
    {
        "Model": "MobileNetV2",
        "Accuracy": 0.7677,
        "Precision": 0.7362,
        "Sensitivity": 0.7358,
        "Specificity": 0.9231,
        "F1 Score": 0.7350,
        "ROC-AUC": 0.9426
    },
    {
        "Model": "EfficientNetB0",
        "Accuracy": 0.7743,
        "Precision": 0.7473,
        "Sensitivity": 0.7495,
        "Specificity": 0.9256,
        "F1 Score": 0.7480,
        "ROC-AUC": 0.9462
    }
])


# ============================================================
# SAVE CSV
# ============================================================

results.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)


# ============================================================
# PRINT TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Sensitivity": "{:.4f}".format,
            "Specificity": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model = results.loc[
    results["Accuracy"].idxmax()
]

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(f"Model:       {best_model['Model']}")
print(f"Accuracy:    {best_model['Accuracy']:.4f}")
print(f"F1 Score:    {best_model['F1 Score']:.4f}")
print(f"ROC-AUC:     {best_model['ROC-AUC']:.4f}")


# ============================================================
# ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Test Accuracy Comparison")

plt.ylim(0, 1)

for i, value in enumerate(results["Accuracy"]):
    plt.text(
        i,
        value + 0.02,
        f"{value * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# F1 COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["F1 Score"]
)

plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.title("Test F1 Score Comparison")

plt.ylim(0, 1)

for i, value in enumerate(results["F1 Score"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "f1_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# ROC-AUC COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["ROC-AUC"]
)

plt.ylabel("ROC-AUC")
plt.xlabel("Model")
plt.title("Test ROC-AUC Comparison")

plt.ylim(0, 1)

for i, value in enumerate(results["ROC-AUC"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_auc_comparison.png",
    dpi=300
)

plt.close()


print("\nComparison files saved in:")
print(OUTPUT_DIR)

print("\nDone!")