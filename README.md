# A/B Test: Did the SmartAd Campaign Raise Brand Awareness?

An online ad campaign was run as an A/B test. Users in the **exposed** group saw
the creative ad and users in the **control** group saw a dummy ad. Both groups
were then shown a one-question brand-awareness survey. This project tests whether
the ad changed the share of users who answered "yes", and whether user-level
features can predict that answer.

## Findings

**Exposed users said "yes" 46.9% of the time (308 of 657) vs 45.1% for control
(264 of 586).** The difference is +1.8 percentage points (+4.1% relative), with a
95% confidence interval of **−3.7 to +7.4 pp**. A two-proportion z-test gives
**p = 0.52** (chi-square p = 0.56).

**Recommendation: the test does not show that the ad worked. Don't credit it
with an awareness lift or scale spend based on this result.** The interval runs
from a small negative effect to a moderate positive one, so the test can't tell
which of these is true.

The main problem is sample size. Only 1,243 of the 8,077 users (15%) answered the
survey. Detecting a 10% relative lift (45% → 49.6%) at α = 0.05 and 80% power needs
about 1,930 responders per group, roughly three times what was collected. At the
observed response rate, that means about 12,500 impressions per group for a rerun.

The ML models don't change this conclusion. On the survey responders, every model
scores close to chance (test ROC-AUC 0.51–0.56; see [Model results](#model-results)).
Hour, device, browser, OS and group don't carry enough signal to predict
the answer.

## Data

`AdSmartABdata.csv`: 8,077 rows, one per user, collected 3–10 July 2020. The file
is not committed. Place it at `data/AdSmartABdata.csv` (MD5
`f0f7b45b51609ab640b1707d72c60ba1`).

| Column | Description |
|--------|-------------|
| `auction_id` | Unique user/impression ID |
| `experiment` | `control` (dummy ad) or `exposed` (creative ad) |
| `date`, `hour` | When the ad was served |
| `device_make`, `platform_os`, `browser` | User's device |
| `yes` | 1 if the user answered "yes" to the brand question |
| `no` | 1 if the user answered "no" |

Users with `yes = 0` and `no = 0` didn't answer and are excluded from both the test
and the models. The target `converted` is defined as `yes` among responders.

## Reproducing the results

Tested with Python 3.12. Library versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/ab_test.py        # writes results/ab_test.json
python scripts/train.py          # logs 4 runs to MLflow, writes results/model_metrics.json
mlflow ui --port 5000            # browse the runs at http://localhost:5000
```

The outputs from the run behind this README are committed in
[`results/ab_test.json`](results/ab_test.json) and
[`results/model_metrics.json`](results/model_metrics.json).

## Model results

The models were trained on the 1,243 responders, with a stratified 80/20 split
(994 train / 249 test, `random_state=42`). Features: `experiment`, `hour`,
`device_make`, `platform_os`, `browser`, plus `is_peak_hour` and `hour_bin`, which
are derived from `hour`. MLflow experiment: `ab_test_mlops`.

| Model | Test ROC-AUC | 5-fold CV ROC-AUC | F1 | Accuracy | MLflow run ID |
|-------|-------------:|------------------:|---:|---------:|---------------|
| Logistic regression | 0.534 | 0.538 ± 0.010 | 0.520 | 0.570 | `49dc62421b6740268cbcb297a12be6c8` |
| Decision tree | 0.527 | 0.516 ± 0.031 | 0.155 | 0.562 | `7479dd3fb71d45618c62d51d113c0c26` |
| Random forest | 0.555 | 0.529 ± 0.012 | 0.495 | 0.574 | `2fbc463d4a3d462cbc616262ebfa9d25` |
| XGBoost | 0.513 | 0.505 ± 0.035 | 0.425 | 0.510 | `718eede1901d4d54a7f6049302994d61` |

For reference, always predicting "no" gives 0.54 accuracy on the test set. The
random forest has the highest test AUC and is saved as `models/best_model.pkl`, but
its cross-validation score (0.529) suggests that this lead is noise.

## Methods

`scripts/ab_test.py` contains an `ABTest` class with:

- Two-proportion z-test with a 95% CI on the difference (used for the findings above)
- Chi-square test of independence
- Welch's t-test and Mann-Whitney U, for continuous metrics
- A sequential probability ratio test (SPRT)
- A sample-size calculator for a given baseline rate and relative lift

`scripts/train.py` builds one scikit-learn `Pipeline` per model: median
imputation and scaling for numeric columns, mode imputation and one-hot encoding
for categoricals, then the classifier. It logs parameters, metrics and the fitted
model for each model to MLflow.

## Streamlit app

```bash
streamlit run app.py             # http://localhost:8501
```

Pages: A/B testing on an uploaded CSV, predictions from a trained model in
`models/`, and a data explorer. Run `scripts/train.py` first if you want to use
the prediction page.

### Docker

```bash
docker build -t abtest-mlops .
docker run -p 8501:8501 -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" abtest-mlops

docker-compose up --build        # Streamlit on :8501 and an MLflow server on :5000
```

## Repository layout

```
scripts/
  ab_test.py       statistical tests; running it reproduces the findings
  train.py         model training with MLflow tracking
  evaluate.py      evaluate a saved model on a CSV
  helper.py, plots.py
Notebooks/
  01_EDA.ipynb, 02_AB_Testing.ipynb, 03_ML_Modelling.ipynb
results/           metrics from the run reported above
app.py             Streamlit app
ab-testing-app/    early React front-end prototype (not used by app.py)
Dockerfile, docker-compose.yml, requirements.txt
```

`data/`, `models/` and `mlruns/` are created locally and aren't committed.

## Author

Desmond Onam · [GitHub](https://github.com/Desmondonam) · [LinkedIn](https://www.linkedin.com/in/desmond-onam-b64702175/)
