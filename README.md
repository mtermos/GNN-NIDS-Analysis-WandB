# GNN-Based Intrusion Detection System Analysis

## Overview
This project focuses on analyzing the performance of different Graph Neural Networks (GNNs) architectures for Network Intrusion Detection Systems (NIDS). The analysis is conducted across multiple datasets, evaluating different attack categories to understand graphical patterns and model effectiveness. Implementation using Pytorch Lightning and Weight and Baises.

## Project Structure
```
├── main.py                   # Main file for running experiments
├── pre_processing.ipynb      # Data preprocessing steps
├── prepare_graph_files.ipynb # Graph preparation scripts
├── results_analysis.ipynb    # Result analysis and visualization
├── dataset_properties.ipynb  # Obtaining of dataset properties
├── graph_properties.ipynb    # Obtaining of graph properties
├── concat_properties.py      # Script for merging properties files from different datasets into one file
├── requirements.txt          # Dependencies required for the project
├── src/                      # Source code directory
```

## Usage
1. **Preprocess the Data**:
   Run `pre_processing.ipynb` to clean and prepare the dataset.

2. **Prepare Graph Files**:
   Execute `prepare_graph_files.ipynb` to generate graph structures.

3. **Run Experiments**:
   Use `main.py` to train and evaluate GNN models.

4. **Get Properties**:
   Use `dataset_properties.ipynb` and `graph_properties.ipynb`.

5. **Concat All Properties**:
   Run `concat_properties.py` to combine all properties into one file, that can be used in analyzing results and finding patterns.

6. **Analyze Results**:
   Open `results_analysis.ipynb` to visualize performance metrics.

## Datasets
The project supports multiple NIDS datasets, including:
- CIC-TON IoT
- CIC-IDS 2017
- CIC-Bot IoT
- CCD-INID
- NF-UQ-NIDS
- Edge-IIoT
- NF-CSE-CIC-IDS2018
- X-IIoT

## Contributions
Feel free to contribute by submitting issues or pull requests.

## Contact
For questions or collaborations, reach out to the author at [mtermos@cesi.fr].