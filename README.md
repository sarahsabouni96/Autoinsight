# AutoInsight | Used-Car Price Estimator

AutoInsight is an end-to-end machine learning application for estimating used-car prices. Enter a manufacturer, model, fuel type, engine size, year of manufacture, and mileage in the Dash dashboard to see an estimated price and a 90% prediction interval. Each estimate is saved in the prediction history.

## How the project evolved

I built AutoInsight by extending my original used-car price prediction project. In the first version, I explored the data and compared two implementations of linear regression, one in PyTorch and one in scikit-learn. I evaluated their predictions using mean absolute error (MAE), root mean squared error (RMSE), and R².

In this version, I trained and saved a scikit-learn pipeline that combines preprocessing with the selected estimator. FastAPI uses that same pipeline for new requests, and Dash displays the results. This let me take a car from raw input to a saved prediction without duplicating the training transformations in the application.

## Ask about the model

In the dashboard, type a question under **Ask about the model** and select **Ask analyst**. The analyst matches your question to one of five topics and answers using statistics saved with the fitted model.

| Topic | Example question |
|---|---|
| Training data | How was the dataset split? |
| Accuracy | How accurate is the model? |
| Feature importance | Which features matter most? |
| Prediction intervals | How reliable is the prediction interval? |
| Vehicle segments | Which groups of cars are in the data? |

## Data and evaluation

The dataset contains **50,000 second-hand car records**. The target is the recorded `Price`. The CSV gives no currency, mileage unit, or listing date, so estimates are displayed in dataset price units. Performance on current market listings would require a separate evaluation. [Kaggle dataset by Mohsen Behdani](https://www.kaggle.com/datasets/msnbehdani/mock-dataset-of-second-hand-car-sales)

After removing 12 exact duplicates, I reserved 9,998 records for final testing. Of the remaining records, 31,992 were used for fitting and model selection and 7,998 for calibrating prediction intervals. Preprocessing was fitted inside each cross-validation fold. Model selection used mean absolute error, measured in the same units as the recorded price.

| Model | Mean validation MAE across three folds |
|---|---:|
| Ridge regression, best setting | 4,862.69 |
| Histogram gradient boosting, best setting | 385.54 |

Histogram gradient boosting achieved **MAE 359.16**, **RMSE 619.60**, and **R² 0.9986** on the held-out test set. Always predicting the training median gave a test MAE of 10,461.05. The 90% prediction intervals covered **90.86%** of recorded test prices.

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
