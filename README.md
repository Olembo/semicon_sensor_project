# Semiconductor Sensor Anomaly Detection

A full end-to-end pipeline to detect anomalies in semiconductor manufacturing sensor data. This project covers data cleaning, advanced feature engineering, unsupervised and supervised modeling, threshold- and hyperparameter-tuning, and rich visualizations.

---

## Project Structure

semicon_sensor_project/
├── data/
│ ├── raw/ # raw synthetic data
│ └── processed/ # cleaned & feature CSVs (download externally)
├── models/ # trained model artifact (download externally)
├── notebooks/ # Jupyter workflows
│ └── feature_engineering.ipynb
├── outputs/
│ └── charts/ # heatmap, corr matrix, PR curve, etc.
├── scripts/
│ └── generate_data.py # synthetic data generator
├── requirements.txt # Python dependencies
└── README.md # this file


---

## Setup

1. **Clone this repo**  
   ```bash
   git clone https://github.com/Olembo/semicon_sensor_project.git
   cd semicon_sensor_project

2. **Install dependencies**
      pip install -r requirements.txt

3. **Download processed data & model**

 - Processed CSVs
Download and unzip into data/processed/ from:
https://your-storage.link/semicon_data_archive.zip

- Trained model
Download hgb_final.joblib into models/ from:
https://your-storage.link/hgb_final.joblib](https://github.com/Olembo/semicon_sensor_project/tree/main/models)

**Usage**
**Data generation & cleaning**
- (Optional) Regenerate raw data:
      python scripts/generate_data.py

- Open and run the notebook:
      jupyter notebook notebooks/feature_engineering.ipynb

**Explore & visualize**
- Missingness heatmap
- Feature correlation matrix
- Precision–Recall curve


**Modeling & evaluation**
- IsolationForest baseline
- HistGradientBoostingClassifier with threshold and hyperparameter tuning
- Final model performance metrics

**Key Results**
| Experiment                            | Precision | Recall | F1-Score |
| ------------------------------------- | :-------: | :----: | :------: |
| IsolationForest (5% contamination)    |    0.05   |  0.05  |   0.05   |
| HGB Classifier (prob ≥ 0.06)          |    0.29   |  0.63  |   0.40   |
| HGB after tuning (lr=0.05, leaves=63) |    0.29   |  0.63  |   0.40   |

- Precision–Recall AUC: ~0.51
- Optimal rolling window size: 5 runs

**Next Steps**
1. Define and meet stakeholder success criteria (e.g., Recall ≥ 0.80, Precision ≥ 0.50)
2. Extend feature set (longer-window stats, interaction terms)
3. Evaluate alternative models (e.g., LocalOutlierFactor, ensemble stacking)
4. Deploy the final model in a production pipeline with monitoring
