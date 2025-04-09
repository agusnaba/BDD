import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV
file_path = '../data/OZ_GLCM_Fullprint_ROI.csv'
data = pd.read_csv(file_path)

# Filter columns: Keep 'category', 'Filename', and GLCM features starting from 'Dissimilarity'
columns_to_keep = ['category', 'Filename'] + [col for col in data.columns if col not in ['number', 'duration'] and col != 'category' and col >= 'Dissimilarity']
filtered_data = data.loc[:, columns_to_keep].copy()  # Use .loc and .copy() to avoid SettingWithCopyWarning

# Display the filtered data
print("Filtered Data (First 5 Rows):")
print(filtered_data.head())

# Encode the 'category' column to numeric values
label_encoder = LabelEncoder()
filtered_data['category_encoded'] = label_encoder.fit_transform(filtered_data['category'])

# Extract GLCM feature columns (numeric only, excluding 'category' and 'Filename')
features = filtered_data.drop(columns=['category', 'category_encoded', 'Filename']).select_dtypes(include=[float, int])

# Preprocessing: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Perform PCA
pca = PCA(n_components=3)  # Taking 3 principal components
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['category'] = filtered_data['category'].values  # Add original category labels

# Ensure 'Filename' column exists and has the correct length
if 'Filename' in filtered_data.columns and len(filtered_data['Filename']) == len(pca_df):
    pca_df['Filename'] = filtered_data['Filename'].values  # Add filenames
else:
    raise ValueError("The 'Filename' column is missing or its length does not match the PCA data.")

# Separate data for 'good' and 'reject' categories
good_data = pca_df[pca_df['category'] == 'good']
reject_data = pca_df[pca_df['category'] == 'reject']

# Select points for labeling
reject_point = reject_data.iloc[0]  # First 'reject' point
good_cluster_point = good_data.iloc[0]  # First 'good' point from the main cluster
good_outliers = good_data.nlargest(2, 'PC1')  # Two 'good' outliers based on PC1

# Plot the PCA result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points with 'good' label without text
ax.scatter(good_data['PC1'], good_data['PC2'], good_data['PC3'], alpha=0.7, c='blue', label='good')

# Plot points with 'reject' label without text
ax.scatter(reject_data['PC1'], reject_data['PC2'], reject_data['PC3'], alpha=0.7, c='red', label='reject')

# Highlight and label the selected points
ax.text(reject_point['PC1'], reject_point['PC2'], reject_point['PC3'], reject_point['Filename'], color='red', fontsize=10)
ax.text(good_cluster_point['PC1'], good_cluster_point['PC2'], good_cluster_point['PC3'], good_cluster_point['Filename'], color='blue', fontsize=10)
for _, outlier in good_outliers.iterrows():
    ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], outlier['Filename'], color='green', fontsize=10)

# Add legend for 'good' and 'reject'
ax.legend()

ax.set_title('3D PCA Result with Selected Points Labeled')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()