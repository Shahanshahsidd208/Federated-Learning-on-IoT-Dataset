# Federated Learning for IoT Anomaly Detection

## Problem Statement
As IoT devices proliferate, securing these devices and detecting network anomalies become paramount. Traditional methods require transferring large amounts of personal data, raising privacy concerns. This project leverages **Federated Learning (FL)** with deep auto-encoder models to detect anomalies in a decentralized manner, ensuring data privacy while maintaining model performance. Our method improves detection by combining techniques such as **client selection**, **retraining**, and **FedAvgM aggregation**.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Anomaly Detection Process](#anomaly-detection-process)
- [Methods](#methods)
  - [Client Selection](#client-selection)
  - [Retraining Process](#retraining-process)
- [Resources](#resources)
  - [Hardware & Software](#hardware--software)
  - [Datasets & Tutorials](#datasets--tutorials)
- [Results](#results)
  - [FedAvg vs FedAvgM](#fedavg-vs-fedavgm)
  - [Retraining Impact](#retraining-impact)
  - [FL vs Non-FL Comparison](#fl-vs-non-fl-comparison)
- [Conclusion](#conclusion)

## Overview
This project implements a Federated Learning-based approach using deep auto-encoders for IoT anomaly detection. The focus is on maintaining privacy, optimizing training processes with client selection and retraining, and using the **FedAvgM** aggregation technique to ensure the generalization of the model across heterogeneous IoT devices.

## System Architecture
The system comprises client-side model training on IoT devices, with **Federated Learning** to aggregate global updates via the **FedAvgM** algorithm. A **retraining** process ensures model robustness, particularly when dealing with non-IID datasets. This decentralized learning system is scalable and focuses on privacy preservation.

Key Components:
- **Federated Learning**: Aggregation of model parameters from individual IoT devices.
- **FedAvgM Algorithm**: Improves model performance by addressing heterogeneity.
- **Retraining**: Additional training to handle non-IID data.
- **Client Selection**: Random partial device selection for efficient communication rounds.

## Anomaly Detection Process
1. **Local Training**: Each IoT device trains a local deep auto-encoder on network traffic data.
2. **Model Updates**: The weights of each model are sent to the global server for aggregation.
3. **Federated Aggregation**: The global server uses **FedAvgM** or **FedAvg** to combine model updates.
4. **Retraining & Selection**: Models are retrained and partially selected for optimized performance.
5. **Anomaly Detection**: After training, the final model detects anomalies on both the client and new devices.

## Methods

### Client Selection
To improve model efficiency, not all clients participate in every communication round. A subset of devices is randomly selected, minimizing the non-IID dataset impact and reducing communication overhead.

### Retraining Process
Retraining adds robustness by addressing the non-IID nature of client datasets. In each round, a random subset of the benign training data is used for retraining, and the local models are updated before aggregation.

## Resources

### Hardware & Software
Experiments were run on Google Colab with the following setup:

**Hardware**:
- **CPU**: Intel(R) Xeon(R) @ 2.30GHz
- **GPU**: Tesla K80
- **Memory**: 12.7 GB
- **Disk**: 69 GB

**Software**:
- Python 3.7.10
- PyTorch 1.8.1
- Numpy 1.19.5
- Pandas 1.1.5
- Scikit-learn 0.22.2

### Datasets & Tutorials
- **Kaggle Dataset**: [N-BaIoT](https://www.kaggle.com/mkashifn/nbaiot-dataset)
- **Federated Learning Tutorial**: [Udacity FL Course](https://classroom.udacity.com/courses/ud185)
- **PyTorch Tutorial**: [PyTorch Official](https://pytorch.org/tutorials/)

## Results

### FedAvg vs FedAvgM
A comparison between **FedAvg** and **FedAvgM** with retraining (FRN vs MRN) shows that **FedAvgM** outperforms **FedAvg**, particularly in terms of **FPR** and **F1 score**, making it more suitable for our anomaly detection task.

| Method | Avg TPR | Avg FPR | Avg F1 Score | Time (mins) |
|--------|---------|---------|--------------|-------------|
| FRN    | 0.9009  | 0.03498 | 93.073       | 41.7        |
| MRN    | 0.9009  | 0.02189 | 93.096       | 48.0        |

### Retraining Impact
Comparing **FedAvgM** with retraining (MRN) vs without retraining (MNN), the performance metrics remain similar, but **retraining** adds time overhead, suggesting that **no retraining** is more efficient.

| Method | Avg TPR | Avg FPR | Avg F1 Score | Time (mins) |
|--------|---------|---------|--------------|-------------|
| MRN    | 0.9009  | 0.02189 | 93.096       | 48.0        |
| MNN    | 0.9009  | 0.02189 | 93.096       | 36.4        |

### FL vs Non-FL Comparison
The **Federated Learning (FL)** model significantly outperforms the **Non-FL** model in both **TPR** and **F1 score**, while also offering faster training times. The FL model's performance in real-time detection tasks is crucial for IoT anomaly detection.

| Model   | Avg TPR | Avg FPR | Avg F1 Score | Time (mins) |
|---------|---------|---------|--------------|-------------|
| FL      | 0.99988 | 0.02836 | 99.966       | 8.6         |
| Non-FL  | 0.90076 | 0.02232 | 93.095       | 3109        |

## Conclusion
This project demonstrates the superiority of Federated Learning over traditional centralized learning in terms of privacy, security, and model performance. Through techniques like client selection and retraining, the FL model becomes more robust against non-IID data and suitable for real-world IoT anomaly detection.

For more details on the code and implementation, feel free to explore the project repository.

