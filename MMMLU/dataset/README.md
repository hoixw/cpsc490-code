# MMMLU-dataset

This contains the code to build the train/test dataset utilised for the MMMLU task. To save space, we do not include the original source data, but provide the download links below. As a fixed seed is used, this will reproduce the exact dataset.

## Obtaining data

We obtain data from two sources:

1. [MMLU](https://huggingface.co/datasets/cais/mmlu): We use the original MMLU (Massive Multitask Language Understanding) dataset for English. 

2. [MMMLU](https://huggingface.co/datasets/openai/MMMLU): The MMMLU (Multilingual MMLU) dataset contains the MMLU test set translated into 14 other languages. We utilise the following languages: Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Japanese, and Chinese (Simplified). The relevant `.csv` files can be downloaded from the 'Data Studio' section.

As the MMMLU dataset and MMLU dataset are formatted slightly differently (despite having the same questions), we provide `conv.py`, located in the `data/` folder. After downloading the MMLU test dataset [here](https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet?download=true), run `conv.py` to convert the `parquet` file into a `csv`. 

## Producing the train/test Dataset

Run `split.py` to re-produce the train/test dataset. The file expects the original data to be in the `data/` folder, and will put the data in `splits/`. 

I train the model on 1200 prompts for each language, utilising the *same prompts* for each language (albeit, of course, translated into the respective language). For testing, I use 300 *random prompts* for each language. This makes the classification task more difficult.