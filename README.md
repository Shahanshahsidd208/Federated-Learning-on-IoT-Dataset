# Federated Learning for IoT Anomaly Detection

## Introduction
With the rapid increase of IoT (Internet of Things) devices, protecting them from cyber threats and detecting unusual network behavior has become crucial. Traditional methods for anomaly detection require collecting data from these devices, raising concerns about privacy and data security. Federated Learning (FL) solves this issue by allowing each device to train its model locally and share only the model updates, not the data itself. This project applies FL to build an anomaly detection system for IoT devices, ensuring privacy, fast model training, and improved performance across different devices.

## Problem Statement
IoT devices are vulnerable to network attacks, but detecting these anomalies while preserving privacy is a challenge. Traditional methods require data sharing, which can expose sensitive information. This project explores using **Federated Learning** to develop an anomaly detection model that maintains privacy and ensures high performance across various IoT devices.

## Table of Contents
- [Motivation](#motivation)
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
  - [Impact of Retraining](#impact-of-retraining)
  - [FL vs Non-FL Comparison](#fl-vs-non-fl-comparison)
- [Conclusion](#conclusion)

## Motivation
Security in IoT devices is critical, as these devices often handle sensitive data, such as camera footage, home automation systems, and personal health information. Traditional centralized models for anomaly detection require data to be uploaded to a server, risking data privacy and security breaches. **Federated Learning** offers a solution by keeping data on the device and only sharing model updates, allowing privacy-preserving training while still building an effective anomaly detection model. This project is motivated by the need for privacy, security, and efficient anomaly detection in IoT environments.

## System Architecture
The system is designed using Federated Learning (FL), where local models are trained on IoT devices and aggregated on a central server to create a global model. Each device shares only its model parameters, protecting the underlying data. The aggregation method used is **FedAvgM**, which is an improvement over the basic FedAvg algorithm. Additionally, techniques like **retraining** and **partial client selection** are applied to improve model robustness and efficiency, especially with non-IID (non-independent, identically distributed) data.

Key Components:
- **Local Training**: Each IoT device trains its model on its local data.
- **Model Updates**: Devices send their model updates (weights) to the global server.
- **Federated Aggregation**: The global server aggregates updates using FedAvgM to produce a more general model.
- **Retraining & Client Selection**: Further improvement is made by retraining models and selecting a subset of devices for each round of communication.

## Anomaly Detection Process
1. **Local Training**: IoT devices train local models using deep auto-encoders on their own network data.
2. **Model Updates**: Only the model parameters are shared with the global server.
3. **Global Aggregation**: The global model is updated with the new parameters using the **FedAvgM** algorithm.
4. **Retraining & Selection**: Retraining helps refine the model's performance, and client selection ensures efficient updates.
5. **Anomaly Detection**: The trained model detects anomalies, improving with every communication round.

## Methods

### Client Selection
Not all devices are involved in every communication round. A subset is selected randomly, which helps improve training speed and reduce communication overhead. This random selection also ensures the model can generalize better across devices.

### Retraining Process
To handle non-IID data (when data across devices is different), retraining is done periodically to improve the model's accuracy and robustness. Each client retrains its model on new or selected local data, and the updates are incorporated into the global model during the aggregation process.

## Resources

### Hardware & Software
The experiments were conducted using a virtual environment on Google Colab, with the following setup:

**Hardware**:
- **CPU**: Intel(R) Xeon(R) @ 2.30GHz
- **GPU**: Tesla K80
- **Memory**: 12.7 GB RAM
- **Disk**: 69 GB storage

**Software**:
- Python 3.7.10
- PyTorch 1.8.1
- Numpy 1.19.5
- Pandas 1.1.5
- Scikit-learn 0.22.2

### Datasets & Tutorials
The data used for training came from the **N-BaIoT dataset**, which includes IoT network traffic data.

- **Dataset**: [N-BaIoT Dataset on Kaggle](https://www.kaggle.com/mkashifn/nbaiot-dataset)
- **Federated Learning Tutorial**: [Udacity FL Course](https://classroom.udacity.com/courses/ud185)
- **PyTorch Tutorial**: [PyTorch Official](https://pytorch.org/tutorials/)

## Results

### FedAvg vs FedAvgM
Comparing **FedAvg** with **FedAvgM** (both with retraining) shows that **FedAvgM** performs better, particularly in reducing false positives (lower FPR) and achieving a higher F1 score, indicating better anomaly detection.

| Method | Avg TPR | Avg FPR | Avg F1 Score | Time (mins) |
|--------|---------|---------|--------------|-------------|
| FRN    | 0.9009  | 0.03498 | 93.073       | 41.7        |
| MRN    | 0.9009  | 0.02189 | 93.096       | 48.0        |

### Impact of Retraining
By comparing **FedAvgM** with and without retraining (MRN vs MNN), we found that retraining has a minimal impact on the final performance metrics. However, skipping retraining reduces the training time significantly, making it more efficient.

| Method | Avg TPR | Avg FPR | Avg F1 Score | Time (mins) |
|--------|---------|---------|--------------|-------------|
| MRN    | 0.9009  | 0.02189 | 93.096       | 48.0        |
| MNN    | 0.9009  | 0.02189 | 93.096       | 36.4        |

### FL vs Non-FL Comparison
The Federated Learning model (FL) outperforms the Non-FL model in terms of accuracy and speed. The FL model's training is faster and offers better privacy, as no raw data is shared between devices.

| Model   | Avg TPR | Avg FPR | Avg F1 Score | Time (mins) |
|---------|---------|---------|--------------|-------------|
| FL      | 0.99988 | 0.02836 | 99.966       | 8.6         |
| Non-FL  | 0.90076 | 0.02232 | 93.095       | 3109        |

## Conclusion
The Federated Learning approach provides a robust and secure solution for anomaly detection in IoT devices. By keeping the data on devices and only sharing model updates, FL ensures privacy and security while improving detection performance. Through client selection, retraining, and the use of **FedAvgM**, the model is efficient and generalizes well across multiple devices.

Explore the code and further details in the repository for more insights.

