import os
import json

NOTEBOOKS_DIR = 'notebooks'
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

# 1. 01_EDA.ipynb
cells_01 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01. Exploratory Data Analysis (EDA)\n",
            "## Intelligent Loan Risk Assessment & Prediction System\n",
            "This notebook performs exploratory analysis on raw loan application data (`loan_data.csv`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "df = pd.read_csv('../data/raw/loan_data.csv')\n",
            "print('Dataset Shape:', df.shape)\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.info()\n",
            "print('\\nMissing Values:\\n', df.isnull().sum())\n",
            "print('\\nTarget Distribution:\\n', df['Loan_Status'].value_counts(normalize=True))"
        ]
    }
]
create_notebook('01_EDA.ipynb', cells_01)

# 2. 02_Data_Preprocessing.ipynb
cells_02 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02. Data Preprocessing & Feature Engineering\n",
            "This notebook builds domain-specific feature engineering pipelines and Scikit-learn ColumnTransformer."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, os\n",
            "sys.path.insert(0, '../')\n",
            "from src.data.preprocessing import load_raw_data, prepare_dataset, build_preprocessor_pipeline\n",
            "\n",
            "df = load_raw_data('../data/raw/loan_data.csv')\n",
            "X, y = prepare_dataset(df, is_training=True)\n",
            "preprocessor = build_preprocessor_pipeline()\n",
            "X_trans = preprocessor.fit_transform(X)\n",
            "print('Processed Feature Matrix Shape:', X_trans.shape)"
        ]
    }
]
create_notebook('02_Data_Preprocessing.ipynb', cells_02)

# 3. 03_ML_Model_Training.ipynb
cells_03 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03. Machine Learning Model Training & Hyperparameter Tuning\n",
            "Train 7 Machine Learning algorithms and perform RandomizedSearchCV hyperparameter tuning."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.insert(0, '../')\n",
            "from src.ml.train_models import train_and_evaluate_all\n",
            "results_df = train_and_evaluate_all()\n",
            "results_df"
        ]
    }
]
create_notebook('03_ML_Model_Training.ipynb', cells_03)

# 4. 04_Model_Comparison.ipynb
cells_04 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 04. Model Comparison & Empirical Evaluation\n",
            "Compare all trained ML models against Deep Learning ANN using F1, ROC-AUC, Precision, Recall."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "df_comp = pd.read_csv('../reports/model_comparison.csv')\n",
            "df_comp"
        ]
    }
]
create_notebook('04_Model_Comparison.ipynb', cells_04)

# 5. 05_Deep_Learning_Model.ipynb
cells_05 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 05. Deep Learning Neural Network (ANN)\n",
            "Train an Artificial Neural Network with Dense, ReLU, Dropout, and EarlyStopping."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.insert(0, '../')\n",
            "from src.dl.train_nn import train_dl_model\n",
            "df_all = train_dl_model()\n",
            "df_all"
        ]
    }
]
create_notebook('05_Deep_Learning_Model.ipynb', cells_05)

print("All 5 Jupyter Notebooks generated successfully!")
