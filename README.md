# CMPE-255-TimeGPT-Tabula9-Relational-Deep-learning

This repository contains Google Colab notebooks and supporting files for running various machine learning and time series forecasting tasks using TimeGPT, Tabular, and RDL-Relbench. The notebooks cover multivariate time series forecasting, fine-tuning, anomaly detection, and synthetic data generation. Each notebook is documented and includes code explanations to facilitate understanding of the workflow.

## Table of Contents
- [Project Overview](#project-overview)
- [Notebook Descriptions](#notebook-descriptions)
  - [TimeGPT](#timegpt)
  - [Tabular](#tabular)
  - [RDL and Relbench](#rdl-and-relbench)
  
## Project Overview

This project demonstrates the capabilities of TimeGPT, Tabular, and RDL-Relbench, showcasing practical applications in forecasting, anomaly detection, data synthesis, and tabular prediction tasks using graph neural networks (GNNs).

- **TimeGPT**: Provides tools for long-horizon time series forecasting, anomaly detection, and fine-tuning with specific datasets.
- **Tabular**: Focuses on synthetic data generation and zero-shot inference.
- **RDL-Relbench**: Trains GNN-based models for tabular prediction tasks, demonstrating effective relational learning.

Each notebook runs a unique task, with output results displayed directly within Google Colab. The completed notebooks and artifacts are checked into this GitHub repository for reference.

### Prerequisites
1. **Python 3.8+**
2. **Google Colab** account (for running and demonstrating the notebooks)
3. **GitHub** account (for accessing the repository and storing artifacts)
4. 
## Notebook Descriptions

### TimeGPT
1. **TimeGPT Multivariate Forecasting and Long-Horizon Forecasting**  
   *Notebook*: `TimeGPT_Multivariate_LongHorizon_Forecast.ipynb`  
   **Description**: Demonstrates multivariate time series forecasting with a long-horizon perspective. Uses sample time series data and visualizes forecasted results.
   **Reference**: [TimeGPT Multivariate Forecasting](https://docs.nixtla.io/docs/tutorials-multiple_series_forecasting)

2. **Fine-Tuning TimeGPT with Custom Data**  
   *Notebook*: `TimeGPT_FineTune_CustomData.ipynb`  
   **Description**: Shows how to fine-tune TimeGPT using custom time series data, adapting the model to specific forecasting needs.
   **Reference**: [Fine-Tuning with TimeGPT](https://docs.nixtla.io/docs/tutorials-fine_tuning)

3. **Anomaly Detection with TimeGPT**  
   *Notebook*: `TimeGPT_AnomalyDetection.ipynb`  
   **Description**: Implements anomaly detection on time series data using TimeGPT, highlighting unusual patterns.
   **Reference**: [Anomaly Detection](https://docs.nixtla.io/docs/tutorials-anomaly_detection)

4. **Energy Demand Forecasting Using TimeGPT**  
   *Notebook*: `TimeGPT_EnergyForecasting.ipynb`  
   **Description**: Applies TimeGPT to forecast energy demand, showcasing real-world forecasting applications.
   **Reference**: [Energy Forecasting](https://docs.nixtla.io/docs/use-cases-forecasting_energy_demand)

5. **Bitcoin Price Prediction with TimeGPT**  
   *Notebook*: `TimeGPT_BitcoinForecasting.ipynb`  
   **Description**: Utilizes TimeGPT to predict Bitcoin prices based on historical data, demonstrating financial time series forecasting.
   **Reference**: [Bitcoin Forecasting](https://docs.nixtla.io/docs/use-cases-bitcoin_price_prediction)

### Tabular
1. **Synthetic Data Generation**  
   *Notebook*: `Tabular_SyntheticData_Generation.ipynb`  
   **Description**: Uses Tabular to generate synthetic data for a real dataset, providing a practical data augmentation tool.
   **Reference**: [Tabular Synthetic Data](https://github.com/zhao-zilong/Tabula/blob/main/Tabula_on_insurance_dataset.ipynb)

2. **Zero-Shot Inference on Tabular Model**  
   *Notebook*: `Tabular_ZeroShot_Inference.ipynb`  
   **Description**: Demonstrates zero-shot inference capabilities on a pre-trained Tabular model.
   **Reference**: [Zero-Shot Inference](https://github.com/mlfoundations/rtfm/blob/main/notebooks/inference.ipynb)

### RDL and Relbench
1. **GNN-Based Model for Tabular Prediction Task**  
   *Notebook*: `RDL_Relbench_GNN_Tabular_Prediction.ipynb`  
   **Description**: Uses Relbench to train a GNN-based model for tabular prediction tasks, displaying metrics to evaluate performance.
   **Reference**: [Relbench Tutorial](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/train_model.ipynb)
