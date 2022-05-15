# WB2022 Final Project 

**Authors**: Dawid Płudowski, Julia Kaznowska, Maciej Paczóski

## Topic
 Explaining models that predict median house price for households in Califormia

## Data
```
https://www.kaggle.com/camnugent/california-housing-prices
```

## File structure

```
├── notebooks - all notebooks (.ipynb)
├── presentation - presentation files (.html, .ipynb)
│   └── resources
├── README.md
├── requirements.txt
└── resources
    ├── data - unprocessed and processed data (.csv)
    └── models - models saved in .pkl format
```

## Usage

* to create python venv use:

```
pip install -r requirements.txt
```

* to use `Black` auto-formatter paste at the beginning on the notebook:


```
%load_ext lab_black
```

* do not upload `.csv` files to repository;