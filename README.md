# AutoInsight | Used-Car Price Estimator

AutoInsight estimates a car’s recorded price from six inputs: manufacturer, model, fuel type, engine size, year of manufacture, and mileage. Enter a car in the Dash dashboard to see its estimate, a 90% prediction interval, and a history of previous estimates.

## Ask about the model

After estimating a price, you can ask the built-in analyst how the model performed. Type a question in your own words under **Ask about the model** and select **Ask analyst**. Answers come from the saved training and evaluation report. The analyst currently covers five topics:

- **Training data:** “How was the dataset split?”
- **Accuracy:** “How accurate is the model?”
- **Feature importance:** “Which features matter most?”
- **Prediction intervals:** “How reliable is the prediction interval?”
- **Vehicle segments:** “Which groups of cars are in the data?”

For example, asking about interval reliability tells you how often the interval contained the recorded price on held-out cars. In this run, the 90% interval covered about 90.9% of test cars.

## How the project evolved

I started this project with a question: how well could linear regression estimate a used car’s price from its specifications and mileage? In my original Python script, I explored the data, built a preprocessing workflow, and compared two implementations of linear regression: one written in PyTorch and one using scikit-learn. I evaluated their predictions using MAE, RMSE, and R².

I then extended that experiment into an application. The main change was saving the preprocessing steps and the estimator together. When someone enters a car in the dashboard, the API applies the same transformations learned during training. The current version also compares model families, reports uncertainty, and saves each estimate so it can be revisited.

## Data and evaluation

The source contains **50,000 mock second-hand car records**. The target is the recorded `Price`. The CSV gives no currency, mileage unit, or listing date, so estimates are displayed in dataset price units. Performance on current market listings would require a separate evaluation. [Kaggle dataset by Mohsen Behdani](https://www.kaggle.com/datasets/msnbehdani/mock-dataset-of-second-hand-car-sales)

After removing 12 exact duplicates, I reserved 9,998 records for final testing. Of the remaining records, 31,992 were used for fitting and model selection and 7,998 for calibrating prediction intervals. Preprocessing was fitted inside each cross-validation fold. The analyst and dashboard expose the results of this workflow.

| Model | Mean validation MAE across three folds |
|---|---:|
| Ridge regression, best setting | 4,862.69 |
| Histogram gradient boosting, best setting | 385.54 |

Histogram gradient boosting achieved **MAE 359.16**, **RMSE 619.60**, and **R² 0.9986** on the held-out test set. Always predicting the training median gave a test MAE of 10,461.05. The nominal 90% prediction interval contained the recorded price for **90.86%** of test cars.

## Run the project

Download the CSV from the Kaggle link above and place it at `data/car_sales_data.csv`. Run all commands from the folder containing `compose.yaml`.

### With Docker Compose

```sh
docker compose up --build
```

The training service fits and saves the pipeline. FastAPI loads it, Dash provides the interface, and PostgreSQL stores prediction history. Once the services are ready, open the [dashboard](http://localhost:8050) or the [interactive API docs](http://localhost:8000/docs). Use `docker compose down` to stop the services and retain history.

### With Python in VS Code on Windows

In the first PowerShell terminal, create the environment, train the model, and start the API:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m autoinsight.training
.\.venv\Scripts\python.exe -m uvicorn autoinsight.api:app --port 8000
```

Once the API reports “Application startup complete,” leave it running. In a second terminal in the same folder, start Dash:

```powershell
.\.venv\Scripts\python.exe -m autoinsight.dashboard
```

Open the [dashboard](http://localhost:8050). This setup uses SQLite for local history. If the dashboard says “Prediction service unavailable,” check the [API health endpoint](http://localhost:8000/health) and confirm the first terminal is still running.

## Try one prediction

In the dashboard, choose **Ford Fiesta**, **Petrol**, year **2015**, mileage **50,000**, and engine size **1.6**. The trained model described above returned **16,322** price units, with a 90% interval of approximately **15,473–17,171**. The estimate appears in the history table. To exercise the same request from the API docs, submit this body to `POST /predict`:

```json
{"manufacturer":"Ford","model":"Fiesta","fuel_type":"Petrol","year":2015,"mileage":50000,"engine_size":1.6}
```

## Where to find the implementation

| Question | Code |
|---|---|
| How are the records validated and split? | [Data preparation](autoinsight/data.py) |
| How are models selected, intervals calibrated, and artifacts saved? | [Training pipeline](autoinsight/training.py) |
| How does a new car become a prediction? | [Inference](autoinsight/inference.py) and [FastAPI endpoints](autoinsight/api.py) |
| Where are earlier estimates stored? | [Prediction history](autoinsight/storage.py) |
| How are prices, feature importance, and model statistics displayed? | [Dash dashboard](autoinsight/dashboard.py) and [local question matcher](autoinsight/analyst.py) |

## Tests and exploration

Run `python -m pytest -q` in the project environment to test data validation, saved-model inference, API history, and dashboard callbacks. The [GitHub Actions workflow](.github/workflows/test.yml) configures PostgreSQL for integration tests. To explore the fitting data, run `python -m autoinsight.explore` and open `artifacts/exploration.html`.
