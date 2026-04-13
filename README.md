## Cite as
The primary, citable version of this work is archived at Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19206263.svg)](https://doi.org/10.5281/zenodo.19206263)

# Gaussian Process Bayesian Optimization for Ag Nanowire Films

This repository contains a Gaussian Process Regression (GPR) based Bayesian Optimization framework for multi-objective optimization of transparent conductive Ag nanowire films.

## Objectives
- Maximize optical transmittance
- Minimize sheet resistance

## Method Overview
The workflow includes:
- Experimental data loading
- Leave-One-Out Cross Validation (LOOCV)
- Gaussian Process surrogate modeling
- Pareto front filtering
- Bayesian Optimization with uncertainty-aware acquisition
- Proposal of new experimental conditions

## Files
- `notebooks/AgNW_GPR_Colab.ipynb` — main notebook
- `data_GPR.csv` — experimental dataset
- `requirements.txt` — dependencies

## How to Run
Open the notebook in Google Colab or Jupyter and run all cells sequentially.

Install dependencies:
```bash
pip install -r requirements.txt
