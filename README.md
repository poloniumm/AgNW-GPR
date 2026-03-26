## Cite as
The primary, citable version of this work is archived at Zenodo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19206263.svg)](https://doi.org/10.5281/zenodo.19206263)

# Code and data for: Multi-objective Bayesian optimization of AgNW spray-coating for transparent conducting electrodes

## Contents
- `my_data_cleaned.csv`: Experimental dataset (95 points) with three input parameters (ultrasonic power, flow rate, PVP concentration) and measured properties (optical transmittance T, sheet resistance Rₛ).
- `GPR_AgNW.py`: Python code implementing GPR-based Bayesian optimization.

## Requirements
- Python 3.8+
- numpy
- pandas
- scikit-learn

## Usage
Run the script to perform one optimization iteration:
```bash
python gpr_optimization_code.py
