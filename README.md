# Suppressing Biased Samples

##### Code for paper "Suppressing Biased Samples for Robust VQA", TMM 2021.

## Prerequisites

- Python 3.7
- PyTorch == 1.4.1

## Quick start:

- Data preprocess

Run following command:

You can use bash `tools/download.sh` to download the data
and then use `bash tools/process.sh ` to process the data

- Dataset

All dataset under the directory of `data`, We utilize the glove embedding, please download the *glove.6b.300d.txt*

- Training

Run command:

`python main.py`

- Eval

Run command:

`python eval.py`
