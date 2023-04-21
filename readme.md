# Finding Tori

## Description
This repository provides code and metadata for our paper, "Finding Tori: Self-supervised Learning for Analyzing Korean Folk Songs"

## Metadata
Metadata is given in ``metadata.csv``.

## Contour Data
The F0 contour extracted by CREPE can be downloaded from Google Drive using ``gdown`` in terminal.  

~~~
gdown 1LotDWyA73I0T9R6CQu9OX3Nrfa2kju8V
mkdir contour_csv
tar –xvzf contour_csv.tar.gz -C ./contour_csv
~~~


## Code
- We used pipenv for virtual environment. The requirements are given in Pipfile. You can install them by running ``pipenv install``.

- You can train a CNN model using ``python3 train.py``. Currently the code is using `wandb` library to log the training


- You can get the result of table 1 in our paper by running ``python3 final_table.py``. 
  - The script gives nDCG and RF classifier accuracy in two results. Including `others` class or not.
  - To get result of Pitch Histogram with 25 bins, run ``python3 final_table.py --use_histogram --resolution=1``
  - To get result of Pitch Histogram with 124 bins, run ``python3 final_table.py --use_histogram --resolution=0.2``
  - To get result of Region-supervised CNN model, run ``python3 final_table.py --code=region-trained``