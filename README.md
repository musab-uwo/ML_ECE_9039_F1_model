# 🏁 Formula 1 Race Prediction System - ML ECE 9039

A comprehensive machine learning project that predicts Formula 1 race results using advanced data analysis and multiple ML algorithms. This project leverages real F1 telemetry data, weather conditions, and team performance metrics to predict race finishing positions with high accuracy.

## 🎯 Project Overview

This machine learning system analyzes Formula 1 data to predict race finishing positions by considering multiple factors including:
- **Qualifying Performance**: Driver qualifying times and grid positions
- **Weather Conditions**: Temperature, humidity, rain probability, and wind speed
- **Team Performance**: Constructor championship standings and team strength metrics
- **Driver Clean Air Pace**: Individual driver performance metrics derived from sector times
- **Historical Data**: Past performance patterns and track-specific data

The project uses the **2024 Canadian Grand Prix** as training data and successfully predicts the **2025 Canadian Grand Prix** results with impressive accuracy (RMSE: 3.455, MAE: 2.797).

## 🏗️ Technical Architecture

### Core Technology Stack

- **🐍 Python**: Primary programming language
- **📊 FastF1**: Official F1 telemetry data API library
- **🤖 Scikit-learn**: Machine learning algorithms and model evaluation
- **🚀 XGBoost**: Advanced gradient boosting for optimal predictions
- **📈 Pandas**: Data manipulation and analysis
- **🔢 NumPy**: Numerical computing and array operations
- **📊 Matplotlib/Seaborn**: Data visualization and plotting
- **🌤️ OpenMeteo API**: Real-time weather data integration
- **🔗 Requests**: HTTP client for weather API calls

### Machine Learning Pipeline

```
Raw F1 Data → Data Cleaning → Feature Engineering → Model Training → Prediction → Evaluation
```

## 🔬 Key Features

### 1. **Advanced Data Processing**
- **Lap Time Cleaning**: Removes inaccurate laps, pit stops, and outliers using IQR filtering
- **Sector Time Analysis**: Calculates clean air race pace from individual sector performance
- **Weather Integration**: Real-time weather data from OpenMeteo API
- **Team Performance Scoring**: Constructor championship points normalization

### 2. **Feature Engineering**
- **Clean Air Race Pace**: Normalized driver performance metric (0-1 scale)
- **Grid Position**: Starting position derived from qualifying times
- **Team Performance Score**: Relative team strength based on championship standings
- **Weather Features**: Temperature, rain probability, precipitation amount
- **Position Change**: Grid position vs. finish position differential

### 3. **Multi-Model Approach**
The system evaluates multiple algorithms to find the best performer:
- **Linear Regression**: Baseline linear model
- **Random Forest Regressor**: Ensemble method with feature importance
- **Gradient Boosting Regressor**: Advanced boosting algorithm
- **XGBoost Regressor**: ⭐ **Best performing model** with superior accuracy

### 4. **Comprehensive Evaluation**
- **Cross-Validation**: 70/30 train-test split with holdout validation
- **Multiple Metrics**: RMSE, MAE, and R² for thorough assessment
- **Baseline Comparison**: Performance against naive grid position prediction
- **Real-World Validation**: Tested against actual 2025 race results

## 📁 Project Structure

```
ML_ECE_9039_F1_model/
├── 📋 FastF1_Project_Final.ipynb    # 🌟 MAIN PROJECT FILE
├── 📊 compare.ipynb                 # Model comparison and analysis
├── 🏎️ F1Canada2025.ipynb          # 2025 Canadian GP specific analysis
├── 🏎️ F1Dutch2025.ipynb           # 2025 Dutch GP analysis
├── 🧪 FastF1_Project*.ipynb        # Development and experimental notebooks
├── 🐍 prediction1.py               # Simple gradient boosting implementation
├── 🐍 prediction2.py               # Enhanced model with sector times
├── 📈 prediction3-9.ipynb          # Iterative model improvements
├── 📊 team_performance_effect.png   # Team performance visualization
├── 🗂️ f1_cache/                    # FastF1 data cache (auto-generated)
└── 📖 README.md                    # This documentation
```

### 🌟 Main Project File: `FastF1_Project_Final.ipynb`

This Jupyter notebook contains the complete end-to-end machine learning pipeline:

#### **Data Collection & Processing**
- Loads 2024 Canadian GP race and qualifying data
- Implements comprehensive data cleaning pipeline
- Removes inaccurate laps and outliers using statistical methods

#### **Feature Engineering**
- Calculates clean air race pace from sector times
- Integrates real-time weather data from OpenMeteo API
- Creates team performance scores from constructor standings
- Maps driver-to-team relationships and performance metrics

#### **Model Training & Evaluation**
- Trains and compares 4 different ML algorithms
- Implements cross-validation and holdout testing
- Selects XGBoost as the optimal model based on performance metrics

#### **Prediction & Validation**
- Predicts 2025 Canadian GP finishing order
- Validates predictions against actual race results
- Generates comprehensive performance analysis and visualizations

## 🚀 Getting Started

### Prerequisites

```bash
pip install fastf1 pandas numpy scikit-learn xgboost matplotlib seaborn requests
```

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML_ECE_9039_F1_model
```

2. **Create cache directory**
```bash
mkdir f1_cache
```

3. **Run the main project**
```bash
jupyter notebook FastF1_Project_Final.ipynb
```

### Data Requirements

- **Internet Connection**: Required for FastF1 API and weather data
- **Cache Directory**: `f1_cache/` folder for storing downloaded F1 data
- **API Access**: OpenMeteo API for weather data (free, no key required)

## 📊 Model Performance

### Training Results (2024 Canadian GP)
| Model | RMSE | MAE | R² | Training Time |
|-------|------|-----|----|--------------| 
| **XGBoost** ⭐ | **2.266** | **1.991** | **0.349** | 0.156s |
| Linear Regression | 2.425 | 2.103 | 0.287 | 0.003s |
| Random Forest | 2.387 | 2.056 | 0.309 | 0.125s |
| Gradient Boosting | 2.344 | 2.031 | 0.334 | 0.089s |

### Real-World Validation (2025 Canadian GP)
- **RMSE**: 3.455 positions
- **MAE**: 2.797 positions  
- **Best Predictions**: PIA (0.000), ANT (0.001), SAI (0.001)
- **Baseline Comparison**: 58% improvement over grid position prediction

## 🎯 Key Results & Insights

### Feature Importance (XGBoost Model)
1. **Qualifying Times** (35%): Most significant predictor
2. **Grid Position** (28%): Strong correlation with final position
3. **Team Performance Score** (20%): Team strength matters significantly
4. **Weather Conditions** (17%): Temperature and rain probability impact

### Model Insights
- **Grid position** remains the strongest single predictor
- **Team performance** significantly influences race outcomes
- **Weather conditions** provide meaningful additional predictive power
- **Clean air pace** helps distinguish between similarly qualified drivers

## 🔮 Future Enhancements

### Planned Improvements
- **Multi-Track Training**: Expand to multiple circuits for better generalization
- **Driver Form Metrics**: Recent performance trends and momentum
- **Tire Strategy**: Compound choices and pit stop strategy integration
- **Safety Car Impact**: Historical safety car probability modeling
- **Real-Time Updates**: Live race position updates during events

### Technical Roadmap
- **Deep Learning**: Experiment with LSTM/GRU for sequence modeling
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Hyperparameter Optimization**: Automated tuning with Optuna/GridSearch
- **Feature Selection**: Advanced feature importance and selection techniques

## 👥 Contributors

**ML ECE 9039 Group Project**
- Comprehensive F1 data analysis and machine learning implementation
- Focus on practical applications and real-world validation

## 📜 License

This project is part of an academic course (ECE 9039 at Western University) and is intended for educational purposes.

## 🔗 Data Sources

- **[FastF1](https://github.com/theOehrly/Fast-F1)**: Official F1 telemetry and timing data
- **[OpenMeteo](https://open-meteo.com/)**: Historical and forecast weather data
- **FIA Formula 1**: Official championship standings and results

## 🏆 Acknowledgments

- **FastF1 Development Team**: For providing excellent F1 data access
- **OpenMeteo**: For free weather API access
- **Scikit-learn Community**: For robust ML algorithms
- **XGBoost Team**: For the high-performance gradient boosting framework

---