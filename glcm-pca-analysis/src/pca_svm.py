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

# Simpan kolom 'Filename' untuk referensi nanti
if 'Filename' in data.columns:
    filenames = data['Filename']
else:
    filenames = None

# Filter columns: Keep 'category' and GLCM features starting from 'Dissimilarity'
columns_to_keep = ['category'] + [col for col in data.columns if col not in ['number', 'duration', 'Filename'] and col != 'category' and col >= 'Dissimilarity']
filtered_data = data[columns_to_keep]
print(filtered_data.columns)
# print("Filtered Data (First 5 Rows):")
# print(filtered_data.head())

# Encode the 'category' column to numeric values
label_encoder = LabelEncoder()
filtered_data['category_encoded'] = label_encoder.fit_transform(filtered_data['category'])

# Extract GLCM feature columns (numeric only, excluding 'category')
features = filtered_data.drop(columns=['category', 'category_encoded','class']).select_dtypes(include=[float, int])
print("\nGLCM Features:")
print(features.head())
if 'category' in filtered_data.columns:
    print("\nLabels (Category):")
    print(filtered_data['category'].head())

# Preprocessing: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['category'] = filtered_data['category'].values
pca_df['category_encoded'] = filtered_data['category_encoded'].values

# Separate data for 'good' and 'reject' categories
good_data = pca_df[pca_df['category'] == 'good']
reject_data = pca_df[pca_df['category'] == 'reject']

# Extract filenames for 'reject' category
if filenames is not None:
    reject_filenames = filenames.loc[reject_data.index]
    # print("\nFilenames for 'reject' category:")
    # print(reject_filenames.tolist())
else:
    print("\nColumn 'Filename' not found in the dataset.")

all_rejects_outliers=[]
# Identify outliers in the 'good' category based on the largest values in 'PC1'
good_outliers = good_data.nlargest(5, 'PC1')  # Adjust the number of outliers as needed
print(good_outliers)

# Identify outliers in the 'reject' category based on the largest values in 'PC1'
reject_outliers_pc1 = reject_data.nlargest(1, 'PC1')  # Adjust the number of outliers as needed
# print(reject_outliers_pc1)
all_rejects_outliers.append(reject_outliers_pc1)

reject_outliers_pc2 = reject_data.nlargest(1, 'PC2')  # Adjust the number of outliers as needed
# print(reject_outliers_pc2)
all_rejects_outliers.append(reject_outliers_pc2)

reject_outliers_pc3 = reject_data.nlargest(1, 'PC3')  # Adjust the number of outliers as needed
# print(reject_outliers_pc3)
all_rejects_outliers.append(reject_outliers_pc3)

reject_outlier = reject_data.nsmallest(1, 'PC1')  # Adjust the number of outliers as needed
all_rejects_outliers.append(reject_outlier)
reject_outlier = reject_data.nsmallest(1, 'PC2')  # Adjust the number of outliers as needed
all_rejects_outliers.append(reject_outlier)
reject_outlier = reject_data.nsmallest(1, 'PC3')  # Adjust the number of outliers as needed
all_rejects_outliers.append(reject_outlier)


# Train an SVM to find the separating hyperplane
svm = SVC(kernel='linear')
svm.fit(pca_df[['PC1', 'PC2', 'PC3']], pca_df['category_encoded'])
w = svm.coef_[0]
b = svm.intercept_[0]
print(f"Hyperplane equation: {w[0]}*PC1 + {w[1]}*PC2 + {w[2]}*PC3 + {b} = 0")

# Create a meshgrid for the hyperplane
xx, yy = np.meshgrid(np.linspace(pca_df['PC1'].min(), pca_df['PC1'].max(), 10),
                     np.linspace(pca_df['PC2'].min(), pca_df['PC2'].max(), 10))
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

# Plot the PCA result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(good_data['PC1'], good_data['PC2'], good_data['PC3'], alpha=0.7, c='blue', label='good')
ax.scatter(reject_data['PC1'], reject_data['PC2'], reject_data['PC3'], alpha=0.7, c='red', label='reject')

# Highlight outliers in the 'good' category
# for _, outlier in good_outliers.iterrows():
#     ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], 'Outlier', color='green', fontsize=10)

# for _, outlier in reject_outliers.iterrows():
#     ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], 'Outlier', color='green', fontsize=10)

# Highlight outliers in the 'reject' category with filenames
for outlier_ in all_rejects_outliers:
    for idx, outlier in outlier_.iterrows():
        filename = filenames.loc[idx] if filenames is not None else "Unknown"
        ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='green', fontsize=10)
# for idx, outlier in reject_outliers_pc2.iterrows():
#     filename = filenames.loc[idx] if filenames is not None else "Unknown"
#     ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='green', fontsize=10)
# for idx, outlier in reject_outliers_pc3.iterrows():
#     filename = filenames.loc[idx] if filenames is not None else "Unknown"
#     ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='green', fontsize=10)


ax.legend()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()