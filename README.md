# Bacterial Growth Curve Analyzer

A Python tool for fitting the modified Gompertz model to bacterial OD600 growth data, extracting key biological parameters, and producing publication-quality plots.

## What it does

Given a CSV of time vs OD600 measurements, the tool:

- Fits the modified Gompertz model (Zwietering et al., 1990) to each condition
- Extracts and reports:
  - K — carrying capacity (maximum OD)
  - mu_max — maximum specific growth rate (per hr)
  - lag — lag phase duration (hr)
  - doubling time (hr), derived as ln(2) / mu_max
- Reports parameter uncertainty
- Produces a plot with raw data, fitted curves, and lag phase markers
- Exports a CSV summary table

## Usage

Install dependencies:

    pip install -r requirements.txt

Generate example data:

    python simulate_data.py

Run the analyzer:

    python growth_analyzer.py --file data/example_data.csv --plot

Save plot and results table:

    python growth_analyzer.py --file data/example_data.csv --save-plot results.png --save-csv results.csv

## Input format

A CSV with a time column and one column per condition:

    time_h, WT, delta_rpoS
    0,      0.05, 0.05
    1,      0.06, 0.05

## The model

The modified Gompertz equation (Zwietering et al., 1990):

    OD(t) = K x exp( -exp( (mu_max x e / K) x (lag - t) + 1 ) )

This produces the classic S-shaped growth curve with lag, exponential, and stationary phases.

## Reference

Zwietering, M.H. et al. (1990). Modeling of the bacterial growth curve. Applied and Environmental Microbiology, 56(6), 1875-1881.

