# Machine Learning Project

This repository contains the implementation of various machine learning models and dimension reduction techniques for the course project.

## Project Structure

*   `data/`: Contains the dataset files.
    *   `reduced/`: Contains datasets after dimension reduction (PCA, tSNE, ISOMAP).
*   `dim-reduction/`: Scripts for performing dimension reduction.
*   `models/`: Implementations of various models (Decision Tree, Ensemble, KNN, Logistic Regression, Naive Bayes, Neural Networks, SVM).
    *   `resnn/`: PyTorch implementation of a Residual MLP.
*   `mylib/`: Custom implementations of Logistic Regression and SVM.
*   `res/`: Results folder containing prediction CSVs and accuracy logs.
*   `visualization/`: Scripts for visualizing data and results.
*   `params/`: Trained model params saved in pickle format.
*   `util.py`: Utility functions for data loading, saving, and testing.
## Reproducing Results

### Important Params
- For offline test (split the original train data into train and test data), specify the train data path with `-f`.
- For online test (submission to kaggle), `-ntr` specifies train data path, `-nte` specifies test data path, `-nsv` specifies result path.

### 1. Dimension Reduction

To generate the reduced datasets used for analysis and training, run the scripts in the `dim-reduction` directory.

**PCA**
Run from the `dim-reduction` directory:
```bash
cd dim-reduction
python PCA.py -c {n_components}
```
This generates `data/reduced/PCA/train-{n_components}.csv` and `test-{n_components}.csv`.

**t-SNE**
Run from the root directory:
```bash
python dim-reduction/tSNE.py -c {n_components} -p {perplexity}
```
This generates `data/reduced/tSNE/train-{n_components}-{perplexity}.csv`.

**Isomap**
Run from the root directory:
```bash
python dim-reduction/isomap.py -c {n_components} -k {n_neighbors}
```
This generates `data/reduced/ISOMAP/train-{n_components}-{n_neighbors}.csv`.

### 2. Offline Test (Local Validation)

To evaluate models locally using a train/test split on the training data.

**Logistic Regression (Custom)**
```bash
python mylib/my-logreg.py -f data/train.csv
```
Or using PCA reduced data:
```bash
python mylib/my-logreg.py -f data/reduced/PCA/train-50.csv
```

**SVM (Custom)**
```bash
python mylib/my-svm.py -f data/train.csv
```

**Random Forest / Ensemble**
```bash
python models/ensemble.py -f data/train.csv
```

**Residual MLP (PyTorch)**
Run from the `models/resnn` directory:
```bash
cd models/resnn
python test_resnn.py
```
This generates prediction file `./resnn_checkpoints/predictions.csv`.

### 3. Online Test (Kaggle Submission)

To train on the full training set and generate predictions for the test set.

**Logistic Regression**
```bash
# sklearn lib
cd models
python logreg.py -ntr ../data/train.csv -nte ../data/test.csv -nsv ../res/LR/pred-logreg.csv
# my implementation
cd mylib
python my-logreg.py -ntr ../data/train.csv -nte ../data/test.csv -nsv ../res/myLR/pred-logreg.csv
```
For PCA embedded data, use the following command
```bash
cd models
python logreg.py -ntr ../data/reduced/PCA/train-{n_components}.csv -nte ../data/reduced/PCA/test-{n_components}.csv -nsv ../res/SVM/pred-svm-pca-{n_components}.csv
```

**SVM**
```bash
# sklearn lib
cd models
python svm.py -ntr ../data/train.csv -nte ../data/test.csv -nsv ../res/SVM/pred-svm.csv
# my implementation
cd mylib
python my-svm.py -ntr ../data/train.csv -nte ../data/test.csv -nsv ../res/mySVM/pred-svm.csv
```
For PCA embedded data, use the following command
```bash
cd models
python svm.py -ntr ../data/reduced/PCA/train-{n_components}.csv -nte ../data/reduced/PCA/test-{n_components}.csv -nsv ../res/SVM/pred-svm-pca-{n_components}.csv
```

**Ensemble**
```bash
cd models
python ensemble.py -ntr ../data/train.csv -nte ../data/test.csv -nsv ../res/RF/pred-rf.csv
```
For PCA embedded data, use the following command
```bash
cd models
python ensemble.py -ntr ../data/reduced/PCA/train-{n_components}.csv -nte ../data/reduced/PCA/test-{n_components}.csv -nsv ../res/RF/pred-rf-pca-{n_components}.csv
```
