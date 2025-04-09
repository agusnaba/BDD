import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV
file_path = '../data/OZ_GLCM_Fullprint_ROI.csv'
data = pd.read_csv(file_path)

# Filter columns: Keep 'category' and GLCM features starting from 'Dissimilarity'
columns_to_keep = ['category'] + [col for col in data.columns if col not in ['number', 'duration', 'Filename'] and col != 'category' and col >= 'Dissimilarity']
filtered_data = data[columns_to_keep]

# Display the filtered data
print("Filtered Data (First 5 Rows):")
print(filtered_data.head())

# Encode the 'category' column to numeric values
label_encoder = LabelEncoder()
filtered_data['category_encoded'] = label_encoder.fit_transform(filtered_data['category'])

# Extract GLCM feature columns (numeric only, excluding 'category')
features = filtered_data.drop(columns=['category', 'category_encoded']).select_dtypes(include=[float, int])

# Display the GLCM feature columns and labels
print("\nGLCM Features:")
print(features.head())
if 'category' in filtered_data.columns:
    print("\nLabels (Category):")
    print(filtered_data['category'].head())

# Preprocessing: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Perform PCA
pca = PCA(n_components=3)  # Taking 3 principal components
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['category'] = filtered_data['category'].values  # Add original category labels
pca_df['category_encoded'] = filtered_data['category_encoded'].values  # Add encoded category labels

# Separate data for 'good' and 'reject' categories
good_data = pca_df[pca_df['category'] == 'good']
reject_data = pca_df[pca_df['category'] == 'reject']

# Train an SVM to find the separating hyperplane
svm = SVC(kernel='linear')
svm.fit(pca_df[['PC1', 'PC2', 'PC3']], pca_df['category_encoded'])

# Get the coefficients of the hyperplane
w = svm.coef_[0]  # Coefficients for the hyperplane
b = svm.intercept_[0]  # Intercept for the hyperplane
print(f"Hyperplane equation: {w[0]}*PC1 + {w[1]}*PC2 + {w[2]}*PC3 + {b} = 0")

# Create a meshgrid for the hyperplane
xx, yy = np.meshgrid(np.linspace(pca_df['PC1'].min(), pca_df['PC1'].max(), 10),
                     np.linspace(pca_df['PC2'].min(), pca_df['PC2'].max(), 10))
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

# Plot the PCA result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points with 'good' label without text
ax.scatter(good_data['PC1'], good_data['PC2'], good_data['PC3'], alpha=0.7, c='blue', label='good')

# Plot points with 'reject' label without text
ax.scatter(reject_data['PC1'], reject_data['PC2'], reject_data['PC3'], alpha=0.7, c='red', label='reject')

# Plot the hyperplane
ax.plot_surface(xx, yy, zz, alpha=0.5, color='green', label='Hyperplane')

# Add legend for 'good' and 'reject'
ax.legend()

# ax.set_title('3D PCA Result with Good and Reject Points and Hyperplane')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()