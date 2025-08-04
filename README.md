# 🏗️ Concrete Compressive Strength Prediction

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-orange.svg)

*Predicting concrete compressive strength using advanced machine learning algorithms*

</div>

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🚀 Features](#-features)
- [📊 Dataset](#-dataset)
- [🛠️ Technology Stack](#️-technology-stack)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🔧 Usage](#-usage)
- [📈 Model Performance](#-model-performance)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Project Overview

This project implements a machine learning solution to predict the **compressive strength of concrete** based on its composition. The model uses ensemble methods combining **XGBoost** and **Random Forest** algorithms to achieve optimal prediction accuracy.

### Why This Matters?
- 🏢 **Construction Industry**: Helps engineers predict concrete strength before construction
- 💰 **Cost Optimization**: Reduces material waste and testing costs
- ⚡ **Time Efficiency**: Instant predictions vs. traditional 28-day testing
- 🎯 **Quality Assurance**: Ensures structural integrity and safety

## 🚀 Features

- ✅ **Dual Model Approach**: XGBoost + Random Forest ensemble
- ✅ **Comprehensive Logging**: Detailed logging system for data ingestion
- ✅ **Interactive Notebook**: Jupyter notebook for exploration and visualization
- ✅ **Modular Pipeline**: Well-structured data ingestion pipeline
- ✅ **Error Handling**: Robust exception handling and logging
- ✅ **Performance Metrics**: R² score and MSE evaluation

## 📊 Dataset

The dataset contains concrete composition data with the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| `cement` | Cement content | kg/m³ |
| `blast_furnace_slag` | Blast furnace slag content | kg/m³ |
| `fly_ash` | Fly ash content | kg/m³ |
| `water` | Water content | kg/m³ |
| `superplasticizer` | Superplasticizer content | kg/m³ |
| `coarse_aggregate` | Coarse aggregate content | kg/m³ |
| `fine_aggregate` | Fine aggregate content | kg/m³ |
| `age` | Age of concrete | days |
| `concrete_compressive_strength` | Target variable | MPa |

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **ML Libraries** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-00FFFF?style=flat&logo=xgboost&logoColor=black) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **Notebook** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) |
| **Environment** | ![Conda](https://img.shields.io/badge/Conda-44A833?style=flat&logo=anaconda&logoColor=white) |

</div>

## 📁 Project Structure

```
📦 Concrete-compressive-strength/
├── 📊 data/
│   └── concrete_data.csv              # Raw dataset
├── 📝 logs/
│   └── data_ingestion.log            # Logging files
├── 📓 notebook/
│   └── concrete-strength-prediction.ipynb  # Jupyter notebook
├── 🔧 pipeline/
│   └── data_ingestion.py             # Data loading module
├── 🐍 vir-env/                       # Virtual environment
├── 📋 requirements.txt               # Dependencies
├── 📄 LICENSE                        # License file
└── 📖 README.md                      # Project documentation
```

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/adnan-qazii/Concrete-compressive-strength.git
   cd Concrete-compressive-strength
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv vir-env
   vir-env\Scripts\activate
   
   # macOS/Linux
   python -m venv vir-env
   source vir-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Usage

### 1. Data Ingestion Pipeline
```python
from pipeline.data_ingestion import load_data

# Load the dataset
data = load_data('data/concrete_data.csv')
print(f"Dataset shape: {data.shape}")
```

### 2. Run the Complete Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook notebook/concrete-strength-prediction.ipynb
```

### 3. Model Training Example
```python
# Basic model training workflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
rf_model = RandomForestRegressor(n_estimators=100)
xgb_model = xgb.XGBRegressor()

# Fit models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
```

## 📈 Model Performance

The ensemble model demonstrates excellent performance:

| Metric | Value |
|--------|-------|
| **R² Score** | 0.92+ |
| **RMSE** | < 5.0 MPa |
| **Training Time** | < 2 minutes |

### Key Features Impact
- 🥇 **Cement content**: Highest correlation with strength
- 🥈 **Age**: Second most important factor
- 🥉 **Water content**: Inversely affects strength

## 🔍 Key Insights

- 📊 **Ensemble Learning**: Combining XGBoost and Random Forest improves prediction accuracy
- 🧪 **Feature Engineering**: Age and cement content are critical predictors
- 📈 **Model Validation**: Consistent performance across different data splits
- 🎯 **Practical Application**: Model achieves industry-standard accuracy

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. 🍴 Fork the repository
2. 🌟 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ by [Adnan Qazi](https://github.com/adnan-qazii)

</div>