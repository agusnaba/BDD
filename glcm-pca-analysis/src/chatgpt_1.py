import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D

# Load your dataset
df = pd.read_csv("../data/OZ_GLCM_Fullprint_ROI.csv")
print(df.columns)
# Define GLCM features and label
glcm_features = [
    'Dissimilarity 0', 'Dissimilarity 45', 'Dissimilarity 90', 'Dissimilarity 135',
    'Energy 0', 'Energy 45', 'Energy 90', 'Energy 135',
    'Homogeneity 0', 'Homogeneity 45', 'Homogeneity 90', 'Homogeneity 135'
]

all_features = ['number', 'Filename', 'category', 'class', 'duration',
       'Dissimilarity 0', 'Dissimilarity 45', 'Dissimilarity 90',
       'Dissimilarity 135', 'Energy 0', 'Energy 45', 'Energy 90', 'Energy 135',
       'Homogeneity 0', 'Homogeneity 45', 'Homogeneity 90', 'Homogeneity 135',
       'Contrast 0', 'Contrast 45', 'Contrast 90', 'Contrast 135',
       'Correlation 0', 'Correlation 45', 'Correlation 90', 'Correlation 135']
glcm_features1 = ['Dissimilarity 0', 'Dissimilarity 45', 'Dissimilarity 90',
       'Dissimilarity 135', 'Energy 0', 'Energy 45', 'Energy 90', 'Energy 135',
       'Homogeneity 0', 'Homogeneity 45', 'Homogeneity 90', 'Homogeneity 135',
       'Contrast 0', 'Contrast 45', 'Contrast 90', 'Contrast 135']
# ,
#        'Correlation 0', 'Correlation 45', 'Correlation 90', 'Correlation 135']

X = df[glcm_features]
y = df['class']  # Ensure 'label' column is present

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 🌐 3D PCA Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=y, cmap='coolwarm', edgecolor='k', s=50)
ax.set_title("3D PCA Scatter Plot of GLCM Features")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
legend = ax.legend(*scatter.legend_elements(), title="Class")
ax.add_artist(legend)
plt.tight_layout()
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.3, random_state=42)

# Apply SMOTE to balance training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

# Train, evaluate, and show confusion matrix for each model
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    # Classification Report
    print(classification_report(y_test, y_pred, target_names=["Good", "Reject"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Reject"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.grid(False)
    plt.show()