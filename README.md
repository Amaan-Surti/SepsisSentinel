# SepsisSentinel

Real-time ICU sepsis early-warning system built on the PhysioNet/Computing in Cardiology Challenge 2019 dataset.

## Overview
SepsisSentinel is an end-to-end clinical machine learning and decision-support pipeline designed to detect sepsis early in ICU patients (using hourly vital signs, lab measurements, and clinical scores).

## Setup
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Citation
Reyna et al., "Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019", *Crit Care Med* 2020;48(2):210–217.
