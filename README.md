# 🏀 NBA Missing Player Prediction

## 📌 Project Objective

This project aims to build a **machine learning model** that can predict the **best missing home team player** for an NBA game segment, given the lineup of four known home players and the full away team lineup. The model uses **historical matchup data**, player-level statistics, and team-level dynamics to determine which player would most likely enhance the home team's performance in that segment.

---

## 🛠️ Project Structure

```
NBA-missing-player/
├── config.py                      # Global configs (paths, years, seed)
├── pipeline.py                    # Orchestrates the data pipeline
├── models/
│   ├── encoder.py                 # Lineup encoding and binary sample generation
│   ├── evaluator.py               # Model evaluation on test data
│   └── trainer.py                 # XGBoost training logic
├── utils/
│   ├── data_utils.py              # I/O utilities for data and encoders
│   ├── feature_utils.py           # Feature generation for players
│   └── stats_utils.py             # Statistical profiling of players
├── input-dataset/                 # Historical matchup data (matchups-YYYY.csv)
├── test-dataset/                  # Test segments and true labels
├── data/                          # Best hyperparameters for XGBoost per year
├── evaluation_results.json        # Model evaluation results
```

---

## 🚀 Setup & Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/PeterWadie/NBA-missing-player.git
cd NBA-missing-player
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Make sure your environment supports:
- `pandas`
- `scikit-learn`
- `xgboost`
- `numpy`

### 3. Prepare Your Data

Ensure your data folders are structured as follows:

```
input-dataset/
  ├── matchups-2007.csv
  ├── matchups-2008.csv
  ...
  └── matchups-2015.csv

test-dataset/
  ├── NBA_test.csv
  └── NBA_test_labels.csv
```

### 4. Run the Pipeline

This command:
- Loads and processes each year’s data
- Computes player stats
- Generates binary classification samples
- Encodes categorical variables
- Trains an XGBoost model per year
- Evaluates all models on the test set

```bash
python pipeline.py
```

---

## 📈 Results Overview

Evaluation accuracy across seasons:

| Year | Accuracy |
|------|----------|
| 2007 | 91%      |
| 2008 | 90%      |
| 2009 | 94%      |
| 2010 | 97%      |
| 2011 | 96%      |
| 2012 | 95%      |
| 2013 | 98%      |
| 2014 | 98%      |
| 2015 | 94%      |
| 2016 | **7%** ❌ (out-of-distribution test year) |
| **Cumulative** | **86%** ✅ |

🔎 **Note**: The performance drops significantly in 2016 due to out-of-sample prediction challenges. The model uses the 2015 version for 2016 testing.

---

## 🧠 Model Architecture

The model is a **binary classifier** trained with **XGBoost** to distinguish between:
- ✅ A player who was part of a successful 5-man lineup (positive label)
- ❌ A candidate player not part of the lineup (negative label)

For each training instance:
- One player is treated as the **candidate**, the rest as present.
- Features capture the synergy between the candidate and others, plus historical win rates and roles.

Each game segment contributes multiple training samples:
- 5 positive (real lineup permutations)
- N negative (roster players not in the lineup)

---

## ⚙️ Data Preprocessing & Feature Engineering

### Step-by-Step Breakdown:

#### ✅ 1. **Matchup Segments**
Each row in the dataset represents a game segment with:
- Home and away team lineups
- Starting minute of the segment
- Outcome (1 = home advantage, -1 = away)

#### ✅ 2. **Binary Expansion**
Convert each segment into multiple **binary candidate samples**:
- One for each actual lineup player (positive)
- Multiple for other roster players not in the lineup (negative)

#### ✅ 3. **Player & Team Encoding**
Use `LabelEncoder` to map player/team names to integers.

#### ✅ 4. **Feature Engineering**

Each sample includes:
- **Player statistics**
  - Overall/home/away win rates
  - Average starting minute
- **Synergy features**
  - Candidate’s synergy with the 4 present players
  - Synergy between the present players themselves
- **Head-to-head**
  - Candidate's historical win rate against each opposing player

---

## 🧪 Model Evaluation

- **Test Set Composition**:
  - One player is missing from the home lineup
  - Model must predict which roster player best completes the lineup
- **Evaluation Metric**: **Accuracy**
- **Process**:
  - Generate feature samples for each possible candidate
  - Score using the year-specific XGBoost model
  - Compare top candidate with the ground truth

---

## 🔮 Prediction Example

Given:
```csv
home_0, home_1, home_2, home_3, home_4, away_0 ... away_4
"A", "B", "C", "D", "?", "E", "F", "G", "H", "I"
```

Model predicts:
> Best candidate = `"E"` (from team roster not already in `"A"`, `"B"`, `"C"`, `"D"`)

---

## 💡 Future Improvements

- Handle **cross-season generalization** better (mitigate 2016 failure)
- Add **position-based** features (guard, center, etc.)
- Use **graph-based models** for better synergy modeling
- Try **neural embeddings** for players

---

## 👨‍🔬 Author

**Peter Wadie**  
GitHub: [@PeterWadie](https://github.com/PeterWadie)

---

## 📃 License

MIT License – feel free to use, modify, and share with attribution.