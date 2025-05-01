# MMLU-Pro Scraper

This contains the code to scrape the necessary `.pcap` files used to train and test the model.

## Scraping data

To scrape the `.pcap` files, simply run `main.py`. The file expects there to be a `data/` folder within this directory that contains `train.csv` and `test.csv`, so those must be copied over. 

## Constants

The following constants at the top of the file will need to be set:

- `MODEL`: to be set to the model name on OpenRouter
- `OPENROUTER_API_KEY`: to be set to the API key. OpenRouter allows charges to be forwarded to an OpenAI / Gemini AI Studio / etc API account, but this must be done on their website and they charge a small fee for doing so.
- `PROVIDER`: to be set to the model provider. Nearly every model has multiple providers. For OpenAI-based models, we always use the `OpenAI` provider. For Llama 3.3, we use `Friendli`.
- `LOCAL_IP`: to be set to the IP of the system.

Some constants may be modified:

- `SCRAPE_TRAIN` and `SCRAPE_TEST`: this tells the model if it should scrape train and test data respectively. Test data is collected in a different VM to the train data in our scenario, but this quick switch allows us to use the same code.
- `NUM_WORKERS`: determines the number of parallel LLM requests. If set to `1`, requests will be serial. An upper bound should be the number of cores on the machine. 