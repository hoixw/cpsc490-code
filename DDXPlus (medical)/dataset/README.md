# DDXPlus (Medical) Dataset

This contains the code to build the train/test dataset utilised for the Medical task. To save space, we do not include the original source data, but provide the download links below. As a fixed seed is used, this will reproduce the exact dataset.

## Obtaining data

The English DDXPlus is available for download at [https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585). The `release_train_patients.csv`, `release_test_patients.csv` and `release_validate_patients.csv` will need to be downloaded and placed in the `data/` folder.

## Producing the train/test Dataset

Run `split.py` to re-produce the train/test dataset as described in the theiss. The file expects the original data to be in the `data/` folder, and will put the data in `split/`. 
I train the model on 1200 random prompts, and for testing, I use 300 random prompts for each pathology. We only utilise the ten most frequent pathologies.