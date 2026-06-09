# Logistic Regression for Feature Importance Calculation

## Overview

This script computes feature importance weights using logistic regression coefficients.

- Handles missing values via mean imputation
- Standardizes features to zero mean and unit variance
- Trains an unregularized logistic regression model
- Computes feature importance as absolute coefficient values
- Normalizes importance scores to obtain weights that sum to 1

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies