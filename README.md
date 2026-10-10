This repository contains several experiments for forecasting building electricity production or consumption. The main prototypes are listed below.

| Prototype | Location | What it does |

| **1 - Random Forest** | [`Friso-Jenjira/RandomForest_Predict_Production/`](Friso-Jenjira/RandomForest_Predict_Production/) | Predicts `production_w` and compares a Random Forest model with a copy-yesterday baseline using a chronological 80/20 split. |

| **2 - Ridge, 15 minutes** | [`Friso-Jenjira/First_Version_Ridge_Regression_15m_Horizon/`](Friso-Jenjira/First_Version_Ridge_Regression_15m_Horizon/) | Predicts `consumption_w` 15 minutes ahead. Compares persistence, copy-yesterday, Ridge, and Ridge with weather. This folder also contains saved models and result charts. |

| **2 extension - Ridge, 36 hours** | [`Friso-Jenjira/Ridge_36h_Experiment/`](Friso-Jenjira/Ridge_36h_Experiment/) | Extends Ridge to 144 forecast steps (36 hours at 15-minute intervals) and evaluates 80/20, 90/10, and walk-forward validation. Results are in `output/experiment_36h/`. |

| **3 - Histogram Gradient Boosting** | [`Kristian/prototype3_histgradientboosting_36h.py`](Kristian/prototype3_histgradientboosting_36h.py) | Tests HGB and HGB with weather against persistence and copy-yesterday over the same 36-hour horizon and validation methods as the Ridge extension. |

| **3 accuracy variant** | [`Joel/prototype3_hgb_36h_accuracy.py`](Joel/prototype3_hgb_36h_accuracy.py) | Trains a separate HGB model for every forecast step, using extra lag and rolling features. It uses an 80/20 chronological split with a 36-hour safety gap. |

| **Lightweight linear demo** | [`Atakan/forecast_consumption.py`](Atakan/forecast_consumption.py) | A dependency-free ordinary least-squares model that trains on three months of hourly data and forecasts consumption up to 36 hours ahead. |

The Envitron CSV data are not included for privacy. To run a prototype, place `electricity.csv`, `weather.csv`, and `buildings.csv` in that prototype's expected `data/` folder and follow its local README or setup guide.

The empty `test.py` files in some personal folders are placeholders and are not working prototypes.
