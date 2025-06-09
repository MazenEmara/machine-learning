# Machine Learning

A collection of university project notebooks and learning code (2021–2023) illustrating a range of supervised and unsupervised learning techniques, evaluation metrics, and neural network architectures. Each notebook covers a distinct algorithm or analysis pipeline, complete with data processing, model training, evaluation, and visualization.

## Table of Contents

1. [Overview](#overview)
2. [Notebooks](#notebooks)
3. [Repository Structure](#repository-structure)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Contact](#contact)

---

## Overview

This repository brings together hands-on implementations of classic and modern machine learning methods, including:

- **Regression & Classification** on digit/image datasets
- **Clustering** and cluster-validation metrics
- **Neural network** and **convolutional neural network** examples
- **Dialect identification** for Arabic text
- Comprehensive **model comparison** and performance analysis

Use these notebooks as learning references, templates for your own projects, or as homework/assignment code examples.

---

## Notebooks

| File Name                                                 | Description                                                                                                                                                    |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Advanced/convolutional_neural_network.ipynb**           | Build and train a Convolutional Neural Network (CNN) for image classification; visualize training history and filter activations.                              |
| **Advanced/DBI_index.ipynb**                              | Compute cluster-validation metrics—Davies–Bouldin and Dunn’s indexes—on clustering results and compare across algorithms.                                      |
| **Advanced/dialect_identification_for_Arabic_text.ipynb** | Perform Arabic text dialect identification using NLP preprocessing, feature extraction (e.g., TF-IDF), and classification (e.g., Logistic Regression).         |
| **Advanced/fuzzy-cmeans.ipynb**                           | Implement the Fuzzy C-Means clustering algorithm; compute membership degrees, update centroids, and plot the objective function for various fuzziness values.  |
| **Mini_Projects/Bias_Impact_Classifier.ipynb**            | Analyze how including an explicit bias term affects one-vs-all linear classifier accuracy and confusion matrices.                                              |
| **Mini_Projects/OneVsAll_Classifier.ipynb**               | Build a one-vs-all linear classifier for digit recognition using the pseudoinverse solution; evaluate via confusion matrix.                                    |
| **Mini_Projects/Regression_Models.ipynb**                 | Compare Linear, Ridge, and Polynomial (degrees 1–6) regression on a digit-image dataset; plot RMSE vs. degree and select the optimal model.                    |
| **Mini_Projects/KNN_sample.ipynb**                        | Demonstrate K-Nearest Neighbors classification on a sample dataset; explore k-value selection and performance metrics.                                         |
| **Mini_Projects/ML_classification.ipynb**                 | General classification pipeline: data loading, feature engineering, model training (e.g., Decision Trees, SVM), and comparative evaluation.                    |
| **Mini_Projects/NeuralNetworks.ipynb**                    | Implement a feed-forward neural network for classification; include data preprocessing, model architecture defined in Keras/TensorFlow, and performance plots. |

---

## Repository Structure

```text
.
├── Advanced/
│   ├── convolutional_neural_network.ipynb
│   ├── DBI_index.ipynb
│   ├── dialect_identification_for_Arabic_text.ipynb
│   └── fuzzy-cmeans.ipynb
├── Mini_Projects/
│   ├── Bias_Impact_Classifier.ipynb
│   ├── OneVsAll_Classifier.ipynb
│   ├── Regression_Models.ipynb
│   ├── KNN_sample.ipynb
│   ├── ML_classification.ipynb
│   └── NeuralNetworks.ipynb
└── README.md
```

---

## Usage

1. **Launch JupyterLab**

   ```bash
   jupyter lab
   ```

2. **Open any notebook** from the file browser.
3. **Run cells sequentially** to reproduce data loading, model training, and plots.
4. **Modify parameters** (e.g., regularization strength, network architecture) to experiment further.

---

## Contact

For questions or feedback, please reach out to:

- **Your Name** – [mazen.emara01@gmail.com](mailto:mazen.emara01@gmail.com)
- GitHub: [https://github.com/your-username](https://github.com/MazenEmara)

---

**Enjoy exploring these machine learning techniques!**
