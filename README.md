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

* to work with gitflow:
  * work only on `feature`/`hotfixes` branches
  * upload to `develop` after finishing `feature`
  * ask others before release to `master`

```
git flow init [start gitflow]
git flow feature start branch_name [start branch]
git flow feature finish branch_name [end branch and merge to develop]
```

* do not upload `.csv` files to repository;