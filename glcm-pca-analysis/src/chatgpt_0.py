import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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
print(y)
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.3, random_state=42)

# Balance data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Classifiers
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=['Good', 'Reject']))
