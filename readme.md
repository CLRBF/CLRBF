This repository contains code for solving Poisson's equation, the Wave equation, and Thermal Stress Analysis, including forward and inverse problems.

## Repository Structure

- `code/`: Source code for all solvers.
  - **Poisson's Equation**: Run forward problem code to generate data, then use it for the inverse problem.
  - **Wave Equation**: Run forward problem code to generate data, then use it for the inverse problem.
  - **Thermal Stress Analysis**: Uses data from `data/` for both forward and inverse problems.
- `data/`: Input data for Thermal Stress Analysis.
- `result/`: Output results from forward problem solvers.

## Usage

1. Install required packages:
   ```bash
   pip install -r requirements.txt
