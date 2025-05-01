# Data Leaks Through Speculative Decoding: Network Side-Channel Attacks on LLM Inference
## CPSC 490 Project — Sachin Thakrar

This repo contains the code necessary to reproduce all experiments discussed within the final report. This includes both the code necessary to reconstruct datasets, and to train any models. This repo does not contain *every* experiment conducted, just the ones within in the final report. 

The `DDXPlus`, `MMLU-Pro-Augment` and `MMMLU` folders contain the code needed to *scrape* the necessary pcap files for that specific experiment. The `model/` folder contains the model code utilised across all experiments. The `website/` folder contains the source code for the project website. 

For more detail, there are `README.md` files in subfolders going into more detail about the scraping and training process.

See `requirements.txt` for a dependency list.