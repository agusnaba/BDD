import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import re
import warnings

# Load data from CSV
file_path = '../data/OZ_GLCM_Fullprint_ROI.csv'
data = pd.read_csv(file_path)

print(data.columns)
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
# print("\nGLCM Features:")
# print(features.head())
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
# Identify outliers in the 'reject' category based on the largest values in 'PC1'
max_reject_pc1 = reject_data.nlargest(1, 'PC1')  # Adjust the number of outliers as needed
all_rejects_outliers.append(max_reject_pc1)
for idx, outlier in max_reject_pc1.iterrows():
    maxpc1=outlier

max_reject_pc2 = reject_data.nlargest(1, 'PC2')  # Adjust the number of outliers as needed
all_rejects_outliers.append(max_reject_pc2)
for idx, outlier in max_reject_pc2.iterrows():
    maxpc2=outlier

max_reject_pc3 = reject_data.nlargest(1, 'PC3')  # Adjust the number of outliers as needed
all_rejects_outliers.append(max_reject_pc3)
for idx, outlier in max_reject_pc2.iterrows():
    maxpc3=outlier

min_reject_pc1 = reject_data.nsmallest(1, 'PC1')  # Adjust the number of outliers as needed
all_rejects_outliers.append(min_reject_pc1)
for idx, outlier in min_reject_pc1.iterrows():
    minpc1=outlier

min_reject_pc2 = reject_data.nsmallest(1, 'PC2')  # Adjust the number of outliers as needed
all_rejects_outliers.append(min_reject_pc2)
for idx, outlier in min_reject_pc2.iterrows():
    minpc2=outlier

min_reject_pc3 = reject_data.nsmallest(1, 'PC3')  # Adjust the number of outliers as needed
all_rejects_outliers.append(min_reject_pc3)
for idx, outlier in min_reject_pc3.iterrows():
    minpc3=outlier


# Train an SVM to find the separating hyperplane
svm = SVC(kernel='linear')
svm.fit(pca_df[['PC1', 'PC2', 'PC3']], pca_df['category_encoded'])
w = svm.coef_[0]
b = svm.intercept_[0]
# print(f"Hyperplane equation: {w[0]}*PC1 + {w[1]}*PC2 + {w[2]}*PC3 + {b} = 0")

# Create a meshgrid for the hyperplane
xx, yy = np.meshgrid(np.linspace(pca_df['PC1'].min(), pca_df['PC1'].max(), 10),
                     np.linspace(pca_df['PC2'].min(), pca_df['PC2'].max(), 10))
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

# Plot the PCA result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# print("Max PC1, PC2, PC3:")
# print(maxpc1['PC1'], maxpc2['PC2'], maxpc3['PC3'])
# print(good_data['PC1'], good_data['PC2'], good_data['PC3'])

tol=0.5
index_inside_reject=[]
index_outside_reject=[]
for idx, row in good_data.iterrows():
    # print(f"Index: {idx}, PC1: {row['PC1']}, PC2: {row['PC2']}, PC3: {row['PC3']}")
    if (row['PC1'] < maxpc1['PC1']+tol and row['PC2'] < maxpc2['PC2']+tol and row['PC3'] < maxpc3['PC3']+tol and  
                row['PC1'] > minpc1['PC1']-tol and row['PC2'] > minpc2['PC2']-tol and row['PC3'] > minpc3['PC3']-tol):
        index_inside_reject.append(idx)
    else:
        index_outside_reject.append(idx)

filtered_good_data = good_data.loc[index_inside_reject]
ax.scatter(filtered_good_data['PC1'], filtered_good_data['PC2'], filtered_good_data['PC3'], alpha=0.7, c='green', label='good-reject?')
filtered_good_data = good_data.loc[index_outside_reject]
# ax.scatter(filtered_good_data['PC1'], filtered_good_data['PC2'], filtered_good_data['PC3'], alpha=0.7, c='blue', label='good')

# ax.scatter(good_data['PC1'], good_data['PC2'], good_data['PC3'], alpha=0.7, c='blue', label='good')
ax.scatter(reject_data['PC1'], reject_data['PC2'], reject_data['PC3'], alpha=0.7, c='red', label='reject')


# Highlight outliers in the 'reject' category with filenames
# i=0
# for outlier_ in all_rejects_outliers:
#     for idx, outlier in outlier_.iterrows():
#         filename = filenames.loc[idx] if filenames is not None else "Unknown"
#         if i<3:
#             ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='green', fontsize=10)
#         else:
#             ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='blue', fontsize=10)
#         i+=1
#         else:
#             ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='blue', fontsize=10)
#         # print(i)
#         i+=1
# for idx, outlier in reject_outliers_pc2.iterrows():
#     filename = filenames.loc[idx] if filenames is not None else "Unknown"
#     ax.text(outlier['PC1'], outlier['PC2'], outlier['PC3'], filename, color='green', fontsize=10)


ax.legend()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()


# for idx in index_inside_reject:
#     filename = filenames.loc[idx] if filenames is not None else "Unknown"
#     print(idx,filename)




source_folder = '../../CAMERA 4_OZ/good_image_png/'
destination_folder = '../../CAMERA 4_OZ/good_reject_fullimage/'

os.makedirs(destination_folder, exist_ok=True)

# for filename in os.listdir(source_folder):
for idx in index_inside_reject:
    filename = filenames.loc[idx] if filenames is not None else "Unknown"
    # Ekstrak angka menggunakan regex
    # match = re.search(r'\d+', filename)
    match=re.findall('[0-9]+', filename)
    # print(match[1])

    # if match:
    #     number = int(match.group())
    #     print(f"Angka yang diekstrak: {number}")
    # else:
    #     print("Tidak ditemukan angka dalam nama file.")

    # print('Namafile:',idx,filename)
    full_file_path = os.path.join(source_folder,f'OZ_good_image{match[1]}.png')
    print('Full path:',full_file_path)
    if os.path.isfile(full_file_path):  # pastikan ini file, bukan folder
        shutil.copy(full_file_path, destination_folder)
        print(f"Copied: {filename}")



# # Path asal dan tujuan
# source_folder = 'folder_asal/'
# destination_folder = 'folder_tujuan/'

# # Pastikan folder tujuan ada
# os.makedirs(destination_folder, exist_ok=True)

# # Nama file yang ingin disalin
# filename = 'contoh_file.txt'

# # Path lengkap
# source_path = os.path.join(source_folder, filename)
# destination_path = os.path.join(destination_folder, filename)

# # Salin file
# shutil.copy(source_path, destination_path)