import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay  # Tambahkan impor ini
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/OZ_GLCM_Fullprint_ROI.csv")

# Define GLCM features and label
features = [
    'Dissimilarity 0', 'Dissimilarity 45', 'Dissimilarity 90', 'Dissimilarity 135',
    'Energy 0', 'Energy 45', 'Energy 90', 'Energy 135',
    'Homogeneity 0', 'Homogeneity 45', 'Homogeneity 90', 'Homogeneity 135'
]
X = df[features]
y = df['class']
print(df['class'], df['category'])

# Standardize and apply PCA
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=3).fit_transform(X_scaled)

# Split and apply SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.3, random_state=42)
X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)  # Set 'Reject' (1) as the positive class
roc_auc = roc_auc_score(y_test, y_proba)

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba, pos_label=1)  # Set 'Reject' (1) as the positive class
pr_auc = average_precision_score(y_test, y_proba)

# Plot
plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Reject = Positive)')
plt.legend()

# PR
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}', color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Reject = Positive)')
plt.legend()

# Confusion Matrix
# plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])  # Ensure 'Good' (0) is negative, 'Reject' (1) is positive
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good (Negative)", "Reject (Positive)"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Reject = Positive)')
plt.grid(False)
# plt.show()

plt.tight_layout()
plt.show()