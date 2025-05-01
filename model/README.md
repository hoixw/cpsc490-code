# Model

This contains `model.py`, a large file that allows us to train a variety of classifier models, including the stacking classifier utilised in the thesis. 

## How to Use

To train the model, run `model.py`. The file expects there to be a `data/` folder that contains the `.pcap` files (in a test/train split). There are some potential arguments to the file:
- `--model_type`, with options `mlp`, `rf`, `stacking`, `xgb`, `svc`, and `lgbm`. The model used in the run in the report is the `stacking` model, which is a simple stacking classifier combining an multi-layer perceptron (`mlp`) and an XGBoost model (`xgb`). 
- `--use_cached_data`. This will use the cached feature data, rather than scraping the data from the `.pcap` files. This makes it easier to tune the model.
- `--scaler_type`, with options `standard` and `minmax`. We use the `standard` scaler.

