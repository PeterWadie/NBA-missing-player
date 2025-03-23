# NBA Lineup Optimization Using Machine Learning

## Overview

This project leverages historical NBA game data and machine learning algorithms to predict the optimal fifth player for an NBA lineup. Given four players from the home team and a full five-player lineup from the away team, the model recommends the best fifth player from the home team's roster to maximize team performance.

## Project Structure

The project follows a modular, organized approach:

```
nba-lineup-optimization/
├── config.py                      # Configuration constants
├── pipeline.py                    # Main pipeline execution script
├── models/
│   ├── __init__.py
│   ├── encoder.py                 # Data encoding and feature expansion
│   ├── evaluator.py               # Model evaluation logic
│   └── trainer.py                 # Training ML models
└── utils/
    ├── __init__.py
    ├── data_utils.py              # Data loading and saving utilities
    ├── feature_utils.py           # Candidate feature generation
    └── stats_utils.py             # Player statistics computation
```

## Data

- **Training Data**: Historical NBA lineup data with segments from various matches.
- **Testing Data**: Scenarios extracted from game segments where the home team's performance was superior.

### Data Columns

| Column Name      | Description                                       |
|------------------|---------------------------------------------------|
| `game`           | Game identifier                                   |
| `season`         | Season year                                       |
| `home_team`      | Home team's name                                  |
| `away_team`      | Away team's name                                  |
| `starting_min`   | Starting minute of the game segment               |
| `home_0` - `home_4` | Player names in home lineup                       |
| `away_0` - `away_4` | Player names in away lineup                       |
| `outcome`        | Segment outcome (1: home win, -1: away win)       |

## Setup & Installation

### Prerequisites

- Python >= 3.7
- pip package manager

### Installation

Clone the repository:

```bash
git clone https://github.com/your_username/nba-lineup-optimization.git
cd nba-lineup-optimization
```

Install required Python packages:

```bash
pip install pandas numpy sklearn catboost xgboost lightgbm optuna
```

## Usage

Run the entire pipeline with the following command:

```bash
python pipeline.py
```

The pipeline performs the following tasks sequentially for each year in the configured range:

1. **Player statistics calculation**
2. **Data expansion into binary samples**
3. **Feature encoding**
4. **Model training** (CatBoost, XGBoost, LightGBM)
5. **Model evaluation**

## Model Details

The pipeline trains three advanced gradient-boosting models:

- **LightGBM**
- **XGBoost**
- **CatBoost**

Hyperparameter tuning is automated using Optuna.

## Results

After execution, evaluation results are stored as JSON files within each year's directory:

```
./{year}/model_comparison_results.json
```

Example of results:

```json
{
  "LightGBM Accuracy": "77.53%",
  "XGBoost Accuracy": "76.98%",
  "CatBoost Accuracy": "77.91%"
}
```

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests. For substantial changes, please open an issue first to discuss what you'd like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data source credits (if applicable)
- Libraries and frameworks (Pandas, scikit-learn, Optuna, CatBoost, LightGBM, XGBoost)

---

**Author:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [YourGitHubUsername](https://github.com/YourGitHubUsername)