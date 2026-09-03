# =============================================================================
# predict_nn.py - Deep Learning ANN Model Loader
# =============================================================================
# Loads the trained PyTorch-based Artificial Neural Network (ANN) model
# from the serialized .keras file for inference in the Django application.
# =============================================================================

import os
import sys

# Add project root to Python path for module resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from src.dl.train_nn import LoanRiskANN, ANNWrapper


def load_dl_model(model_path='models/dl/loan_risk_nn.keras'):
    """Loads the serialized ANN wrapper from disk using joblib.
    Returns None if the model file does not exist."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)
