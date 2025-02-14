# Football Match Prediction Model

A machine learning model for predicting football match outcomes using historical data and advanced statistical features.

## 🎯 Overview

This project implements a Random Forest classifier to predict football match outcomes based on historical match data. It incorporates rolling statistics, team performance metrics, and venue-based features to make predictions.

## ✨ Features

- Match outcome prediction (Win/Loss/Draw)
- Rolling performance metrics
- Team form analysis
- Venue impact consideration
- Automated feature engineering
- Head-to-head prediction capabilities
- Custom team name standardization

## 🚀 Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/football-prediction.git](https://github.com/dmann001/Premier-League-Match-Predictor)
cd Premier-League-Match-Predictor-main
```


## 📊 Data Requirements

Your input file (`matches.csv`) should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Match date |
| team | string | Team name |
| opponent | string | Opponent team name |
| venue | string | Match venue (Home/Away) |
| result | string | Match result (W/L/D) |
| gf | int | Goals scored |
| ga | int | Goals conceded |
| sh | int | Total shots |
| sot | int | Shots on target |
| dist | float | Distance covered |
| fk | int | Free kicks |
| pk | int | Penalties scored |
| pkatt | int | Penalties attempted |

## 🏗️ Model Architecture

### Feature Engineering
- Venue encoding
- Opponent encoding
- Time-based features
- Rolling averages (3-match window) for:
  - Goals (scored/conceded)
  - Shots
  - Shots on target
  - Distance covered
  - Set pieces

### Model Parameters
```python
RandomForestClassifier(
    n_estimators=50,
    min_samples_split=10,
    random_state=1
)
```


## Project Inspired by Data Quests
