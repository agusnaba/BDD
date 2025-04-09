# GLCM PCA Analysis

This project analyzes the GLCM (Gray Level Co-occurrence Matrix) features extracted from images. It performs Principal Component Analysis (PCA) to reduce the dimensionality of the feature set and visualizes the results in a 3D plot.

## Project Structure

```
glcm-pca-analysis
├── data
│   └── OZ_All file GLCM Analysis_Fullprint ROI.csv
├── src
│   ├── main.py
│   └── utils
│       └── __init__.py
├── requirements.txt
└── README.md
```

- **data/OZ_All file GLCM Analysis_Fullprint ROI.csv**: Contains the GLCM analysis results, including various feature columns used for PCA analysis.
- **src/main.py**: The main entry point of the application. Loads the CSV data, extracts GLCM feature columns, performs PCA to reduce the data to three principal components, and generates a 3D plot of the PCA results.
- **src/utils/__init__.py**: Contains utility functions for data processing, PCA analysis, and plotting.
- **requirements.txt**: Lists the dependencies required for the project, such as pandas, scikit-learn, and matplotlib.

## Setup Instructions

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

## Running the Analysis

To run the PCA analysis and generate the 3D plot, execute the following command:

```
python src/main.py
```

Ensure that the `OZ_All file GLCM Analysis_Fullprint ROI.csv` file is located in the `data` directory before running the script. 

## License

This project is licensed under the MIT License.