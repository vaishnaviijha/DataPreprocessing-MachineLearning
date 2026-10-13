
# ============================================================
# HANDWRITTEN DIGIT RECOGNITION USING MACHINE LEARNING
# Dataset: MNIST
# Algorithm: SGDClassifier
# Environment: Python, Scikit-learn, VS Code / Jupyter Notebook
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
from scipy.stats import loguniform


# ============================================================
# 2. LOAD THE MNIST DATASET
# ============================================================

# Download the MNIST dataset from OpenML.
# Each image contains 28 x 28 = 784 pixel values.
mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

# X contains image features and y contains digit labels.
X = mnist["data"].astype(np.float32)
y = mnist["target"].astype(np.uint8)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Number of images:", len(X))
print("Number of features per image:", X.shape[1])


# ============================================================
# 3. VISUALIZE SAMPLE DIGITS
# ============================================================

# Convert a row of 784 pixels into a 28 x 28 image.
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")


# Display one sample digit.
some_digit = X[5]

plt.figure(figsize=(4, 4))
plot_digit(some_digit)
plt.title(f"Example Digit: {y[5]}")
plt.show()


# Display another sample digit.
plt.figure(figsize=(4, 4))
plot_digit(X[9])
plt.title(f"Example Digit: {y[9]}")
plt.show()


# Function to display multiple digit images in a grid.
def plot_digits(instances, images_per_row=10):
    images_per_row = min(len(instances), images_per_row)
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Add blank images if required to complete the grid.
    n_empty = n_rows * images_per_row - len(instances)
    padded = np.concatenate(
        [instances, np.zeros((n_empty, 784))],
        axis=0
    )

    image_grid = padded.reshape(
        n_rows, images_per_row, 28, 28
    )

    big_image = image_grid.transpose(
        0, 2, 1, 3
    ).reshape(
        n_rows * 28, images_per_row * 28
    )

    plt.imshow(big_image, cmap=mpl.cm.binary)
    plt.axis("off")


# Display the first 100 MNIST images.
plt.figure(figsize=(10, 10))
plot_digits(X[:100], images_per_row=10)
plt.title("First 100 MNIST Images")
plt.show()


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Use 80% of the data for training and 20% for testing.
# Stratification preserves digit proportions.
# random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. BUILD THE BASELINE SGD CLASSIFIER
# ============================================================

# Pipeline applies StandardScaler before SGDClassifier.
# This ensures the scaler is fitted only on training data
# during model fitting and cross-validation.
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        random_state=42
    ))
])

# Train the baseline model.
start_time = time.time()
baseline.fit(X_train, y_train)
baseline_time = time.time() - start_time

# Predict labels for unseen test images.
y_pred_baseline = baseline.predict(X_test)

# Calculate baseline accuracy.
baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

print("\nBASELINE MODEL")
print("Accuracy:", round(baseline_accuracy, 4))
print("Training time:", round(baseline_time, 2), "seconds")

# Predict the sample digit using the fitted pipeline.
print("Sample digit prediction:", baseline.predict([some_digit])[0])


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

# Three-fold cross-validation evaluates the model on
# three different training/validation splits.
cv_scores = cross_val_score(
    baseline,
    X_train,
    y_train,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

print("\nCROSS-VALIDATION SCORES")
print(cv_scores)
print("Mean CV accuracy:", round(cv_scores.mean(), 4))


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

# Generate predictions using cross-validation.
# This avoids evaluating the model on the same examples
# it used to fit each fold.
y_train_pred = cross_val_predict(
    baseline,
    X_train,
    y_train,
    cv=3,
    n_jobs=-1
)

# Compare actual and predicted digit labels.
cm = confusion_matrix(y_train, y_train_pred)

print("\nCONFUSION MATRIX")
print(cm)

# Display the confusion matrix graphically.
ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot(cmap="Blues", values_format="d")

plt.title("MNIST Confusion Matrix")
plt.show()


# ============================================================
# 8. NORMALIZED CONFUSION MATRIX
# ============================================================

# Normalize each row so values represent proportions.
row_sums = cm.sum(axis=1, keepdims=True)

normalized_cm = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0
)

# Hide the diagonal to focus on classification errors.
errors_cm = normalized_cm.copy()
np.fill_diagonal(errors_cm, 0)

plt.figure(figsize=(8, 8))
plt.imshow(errors_cm, cmap="gray")
plt.title("Normalized Confusion Matrix: Errors Only")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.colorbar()
plt.show()


# ============================================================
# 9. PRECISION, RECALL AND F1-SCORE
# ============================================================

# Calculate cross-validated predictions for every training sample.
# Each sample is predicted by a model that did not train on it.
y_train_pred = cross_val_predict(
    baseline,
    X_train,
    y_train,
    cv=3,
    n_jobs=-1
)

# Calculate a separate score for each digit.
precision = precision_score(
    y_train, y_train_pred,
    average=None,
    zero_division=0
)

recall = recall_score(
    y_train, y_train_pred,
    average=None,
    zero_division=0
)

f1 = f1_score(
    y_train, y_train_pred,
    average=None,
    zero_division=0
)

print("\nCLASSIFICATION METRICS BY DIGIT")

for digit in range(10):
    print(
        f"Digit {digit}: "
        f"Precision={precision[digit]:.4f}, "
        f"Recall={recall[digit]:.4f}, "
        f"F1-score={f1[digit]:.4f}"
    )


# ============================================================
# 10. PRECISION-RECALL CURVE FOR DIGIT 5
# ============================================================

# Obtain decision scores for each digit using cross-validation.
decision_scores = cross_val_predict(
    baseline,
    X_train,
    y_train,
    cv=3,
    method="decision_function",
    n_jobs=-1
)

# Treat digit 5 as the positive class and all other digits
# as the negative class.
target_digit = 5
y_train_5 = (y_train == target_digit)

# Extract the decision scores for digit 5.
digit_5_scores = decision_scores[:, target_digit]

# Calculate precision and recall at different thresholds.
precisions, recalls, thresholds = precision_recall_curve(
    y_train_5,
    digit_5_scores
)

# Find the first threshold that achieves at least 90% precision.
valid_indices = np.where(precisions[:-1] >= 0.90)[0]

if len(valid_indices) > 0:
    index = valid_indices[0]
    threshold_90 = thresholds[index]

    # Predict digit 5 using the selected threshold.
    predictions_90 = digit_5_scores >= threshold_90

    print("\nPRECISION-RECALL RESULTS FOR DIGIT 5")
    print(
        "Precision:",
        precision_score(
            y_train_5,
            predictions_90,
            zero_division=0
        )
    )
    print(
        "Recall:",
        recall_score(
            y_train_5,
            predictions_90,
            zero_division=0
        )
    )

else:
    threshold_90 = None
    print("No threshold achieved at least 90% precision.")

# Plot precision and recall against the threshold.
plt.figure(figsize=(9, 5))
plt.plot(
    thresholds,
    precisions[:-1],
    "b--",
    label="Precision"
)
plt.plot(
    thresholds,
    recalls[:-1],
    "g-",
    label="Recall"
)

if threshold_90 is not None:
    plt.axvline(
        threshold_90,
        color="red",
        linestyle=":",
        label="90% precision threshold"
    )

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold: Digit 5")
plt.grid(True)
plt.legend()
plt.show()

# Plot precision against recall.
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Digit 5")
plt.grid(True)
plt.show()


# ============================================================
# 11. CALIBRATED CLASSIFIER AND ROC-AUC
# ============================================================

# A calibrated classifier provides probability estimates,
# which can be used to calculate ROC curves and ROC-AUC.
# This stage may take some time because it uses cross-validation.

roc_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", CalibratedClassifierCV(
        SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            random_state=42
        ),
        method="sigmoid",
        cv=3
    ))
])

# Fit using training data only.
roc_model.fit(X_train, y_train)

# Obtain predicted probabilities for the test set.
probabilities = roc_model.predict_proba(X_test)

# Convert labels into a binary indicator matrix:
# one column for each digit from 0 through 9.
y_test_binary = label_binarize(
    y_test,
    classes=np.arange(10)
)

# Calculate the micro-average ROC curve.
fpr_micro, tpr_micro, _ = roc_curve(
    y_test_binary.ravel(),
    probabilities.ravel()
)

# Calculate weighted and micro-average ROC-AUC.
auc_weighted = roc_auc_score(
    y_test,
    probabilities,
    multi_class="ovr",
    average="weighted"
)

auc_micro = roc_auc_score(
    y_test_binary,
    probabilities,
    average="micro"
)

print("\nROC-AUC RESULTS")
print("Weighted ROC-AUC:", round(auc_weighted, 4))
print("Micro ROC-AUC:", round(auc_micro, 4))

# Plot the micro-average ROC curve.
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_micro,
    tpr_micro,
    linewidth=2,
    label=f"Micro ROC-AUC = {auc_micro:.3f}"
)

# The diagonal represents random classification.
plt.plot([0, 1], [0, 1], "k--")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# ============================================================
# 12. HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV
# ============================================================

# Use float32 arrays to reduce memory usage.
X_train32 = X_train.astype(np.float32, copy=False)
X_test32 = X_test.astype(np.float32, copy=False)

# Create a pipeline that scales features and trains SGD.
tuning_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    ))
])

# Define the hyperparameter combinations to explore.
parameter_distributions = {
    "clf__loss": [
        "hinge",
        "squared_hinge"
    ],
    "clf__penalty": [
        "l2",
        "elasticnet"
    ],
    "clf__alpha": loguniform(1e-6, 1e-2),
    "clf__tol": loguniform(1e-5, 1e-2),
    "clf__max_iter": [1000]
}

# Use 30% of the training data for faster experimentation.
rng = np.random.RandomState(42)
subset_size = max(
    1000,
    int(0.30 * len(X_train32))
)

subset_indices = rng.choice(
    len(X_train32),
    size=subset_size,
    replace=False
)

X_small = X_train32[subset_indices]
y_small = y_train[subset_indices]

print("\nHYPERPARAMETER TUNING")
print("Samples used for tuning:", len(X_small))

# Search through 24 randomly sampled configurations.
# Each configuration is evaluated using two-fold CV.
search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=parameter_distributions,
    n_iter=24,
    cv=2,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
search.fit(X_small, y_small)
tuning_time = time.time() - start_time

# Store the best fitted pipeline.
best_model = search.best_estimator_

print("Tuning time:", round(tuning_time, 2), "seconds")
print("Best parameters:", search.best_params_)
print("Best CV accuracy:", round(search.best_score_, 4))


# ============================================================
# 13. EVALUATE THE TUNED MODEL
# ============================================================

# The best estimator is already fitted by RandomizedSearchCV.
# Its pipeline automatically scales raw pixel values.
y_pred_best = best_model.predict(X_test32)

# Evaluate on the same held-out test set as the baseline.
best_accuracy = accuracy_score(
    y_test,
    y_pred_best
)

improvement = best_accuracy - baseline_accuracy

print("\nMODEL COMPARISON")
print("Baseline accuracy:", round(baseline_accuracy, 4))
print("Tuned accuracy:", round(best_accuracy, 4))
print("Accuracy improvement:", round(improvement, 4))

# Print precision, recall, F1-score and support for each digit.
print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred_best,
        zero_division=0
    )
)


# ============================================================
# 14. VISUALIZE RANDOM TEST PREDICTIONS
# ============================================================

# Select 16 different test images for inspection.
rng = np.random.default_rng(42)

indices = rng.choice(
    len(X_test32),
    size=16,
    replace=False
)

images = X_test32[indices].reshape(-1, 28, 28)
actual_labels = y_test[indices]
predicted_labels = y_pred_best[indices]

# Green means correct; red means incorrect.
plt.figure(figsize=(9, 9))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.imshow(images[i], cmap="gray")
    ax.axis("off")

    correct = predicted_labels[i] == actual_labels[i]

    ax.set_title(
        f"Pred: {predicted_labels[i]}\nTrue: {actual_labels[i]}",
        color="green" if correct else "red",
        fontsize=10
    )

plt.suptitle(
    "Test Predictions: Green = Correct, Red = Incorrect"
)
plt.tight_layout()
plt.show()


# ============================================================
# 15. VISUALIZE MISCLASSIFIED EXAMPLES
# ============================================================

# Find test images where the prediction is incorrect.
misclassified_indices = np.where(
    y_pred_best != y_test
)[0]

print(
    "\nNumber of misclassified test images:",
    len(misclassified_indices)
)

# Display up to 16 incorrect predictions.
if len(misclassified_indices) > 0:

    selected_errors = misclassified_indices[:16]

    bad_images = X_test32[selected_errors].reshape(-1, 28, 28)
    bad_actual = y_test[selected_errors]
    bad_predicted = y_pred_best[selected_errors]

    plt.figure(figsize=(9, 9))

    for i in range(len(selected_errors)):
        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(bad_images[i], cmap="gray")
        ax.axis("off")

        ax.set_title(
            f"Pred: {bad_predicted[i]}\nTrue: {bad_actual[i]}",
            color="red",
            fontsize=10
        )

    plt.suptitle("Misclassified MNIST Images")
    plt.tight_layout()
    plt.show()

else:
    print("No misclassified examples were found.")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("MNIST DIGIT RECOGNITION - FINAL SUMMARY")
print("=" * 50)

print(f"Total dataset images: {len(X)}")
print(f"Training images: {len(X_train)}")
print(f"Testing images: {len(X_test)}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Tuned accuracy: {best_accuracy:.4f}")
print(f"Weighted ROC-AUC: {auc_weighted:.4f}")
print(f"Micro ROC-AUC: {auc_micro:.4f}")
print(f"Misclassified test images: {len(misclassified_indices)}")

print("\nProject execution completed.")
