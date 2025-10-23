Iris Flower Classification

This repository contains a single script `iris_classification.py` that demonstrates an end-to-end
Iris classification pipeline using scikit-learn.

Prerequisites
- Python 3.8+ installed
- Install required packages:

  pip install -r requirements.txt

Run

  python iris_classification.py

Notes
- The script uses the built-in Iris dataset from scikit-learn, so no external data downloads are required.
- It will display plots (pairplot and heatmap). If running in a headless environment, open the script and
  change `main(show_plots=True)` to `main(show_plots=False)` to save the plots instead of showing them.
