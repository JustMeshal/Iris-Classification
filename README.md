# Iris Species Classification

A machine learning project focused on classifying Iris flower species. This repository contains the raw dataset, exploratory data analysis, data preprocessing steps, and a serialized Random Forest classifier.

## Repository Structure

| File | Description |
|------|-------------|
| `Iris.csv` | The classic Iris dataset containing 150 samples across 3 species. |
| `iris_rf_model.ipynb` | Jupyter Notebook containing data exploration, visualizations, preprocessing, and model training. |
| `iris_rf_model.pkl` | A serialized Random Forest model trained on the dataset, ready for immediate inference. |
| `requirements.txt` | Python dependencies required to run the notebook. |
| `README.md` | Project documentation. |

## Dependencies

The project relies on standard data science libraries. Ensure you have the following installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

## Usage

### 1. Exploratory Data Analysis & Preprocessing
To explore the dataset and view the preprocessing pipeline:
1. Open a terminal in the project directory.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook iris_rf_model.ipynb
   ```
3. The notebook includes feature distribution histograms, correlation heatmaps, pairplots, and the target encoding/scaling steps.

### 2. Model Inference
You can load the pre-trained Random Forest model directly in Python to make predictions on new data without needing to retrain:

```python
import pickle
import numpy as np

# Load the serialized model
with open('iris_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a new sample [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
# Note: For accurate predictions, ensure input data matches the scaling used during training.
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Make a prediction
prediction = model.predict(sample_data)
print(f"Predicted class label: {prediction[0]}")
```

## Dataset Details

The included `Iris.csv` dataset contains 4 numerical features and 1 target variable:

**Features:**
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`

**Target Classes (`Species`):**
- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`
