# 🌸 K-Nearest Neighbors (KNN) Classification on Synthetic Iris-like Dataset
This repository demonstrates how to implement and visualize a K-Nearest Neighbors (KNN) classifier using a **synthetic flower classification dataset** inspired by the famous Iris dataset.

---

## 📁 Files Included

- `species_dataset.csv`: Synthetic dataset of 300 flower samples across 3 species.
- `knn_classifier.py`: Python code implementing KNN classification using Scikit-learn.
- `README.md`: This file.

---

## 🧪 Dataset Details

The dataset contains 300 samples (100 each of Setosa, Versicolor, and Virginica), each with the following features:

- `sepal_length` (cm)
- `sepal_width` (cm)
- `petal_length` (cm)
- `petal_width` (cm)
- `species` (target class)

All measurements are generated using Gaussian distributions to simulate real-world variation.

---

## ⚙️ Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 🧠 Machine Learning Task

We use the **K-Nearest Neighbors (KNN)** algorithm to classify flower species based on their morphological features.

### Steps:
1. **Generate Synthetic Data** (already done in `species_dataset.csv`)
2. **Preprocess Data** (feature scaling, label encoding)
3. **Train/Test Split**
4. **Fit KNN Classifier** using `sklearn.neighbors.KNeighborsClassifier`
5. **Evaluate Model** with accuracy and confusion matrix
6. **Visualize**:
   - Accuracy vs K plot
   - Decision boundaries using PCA-reduced 2D data

---

## 📈 Example Results

- Best accuracy achieved at **K = 5**
- Confusion matrix and decision boundaries show strong classification performance

---

## ▶️ How to Run

```bash
# Clone the repo and navigate
git clone https://github.com/your-username/knn-flower-classification.git
cd knn-flower-classification

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run script
python knn_classifier.py
