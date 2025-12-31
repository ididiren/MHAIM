# MHAIM: Molecular Holographic Atom-Pair Interaction Matrices for Mutagenicity Prediction

This repository contains the implementation of the MHAIM​ (Molecular Holographic Atom-Pair Interaction Matrices) model, a deep learning framework for predicting chemical mutagenicity from 3D molecular structures.

# Overview

Mutagenicity is a critical property in toxicology and drug discovery. The MHAIM model generates a comprehensive molecular representation by encoding multi-channel non-covalent and covalent atom-pair interactions from energy-minimized 3D structures. It utilizes a hybrid architecture combining Convolutional Neural Networks (CNN), Capsule Networks (CapsNet), and a Triplet Attention mechanism to effectively extract predictive features.

# Repository Structure

## The main files in this repository are:

1-csv_to_sdf.py: Converts molecular data from CSV format to 3D SDF files.

2-Segmentation_of_sdf.py: Processes and segments SDF files.

3-MHAIM_generate_11td.py: Generates the 11-dimensional Molecular Holographic Atom-Pair Interaction Matrices (MHAIM) as molecular features.

4-train_cnn_capsnet.py: The main script for training the CNN+CapsNet hybrid model.

5-tsne_original.py: Visualizes the original input features using t-SNE.

6-tsne_after_primary_capsule.py: Visualizes the features extracted by the primary capsule layer using t-SNE.

cnn_capsnet_model.py: Defines the architecture of the CNN and CapsNet model.

triplet_attention.py: Implements the Triplet Attention mechanism.

pytorchtools.py: Contains utility functions for PyTorch training.

ames_7486.csv, ames_dataset2.csv, ames_dataset6.csv: Benchmark datasets for mutagenicity (Ames test).

# Quick Start

Data Preparation: Use 1-csv_to_sdf.pyto convert your molecular data into 3D SDF format.

Feature Generation: Run 3-MHAIM_generate_11td.pyto create MHAIM features from the 3D structures.

Model Training: Execute 4-train_cnn_capsnet.pyto train the prediction model.

Analysis (Optional): Use 5-tsne_original.pyand 6-tsne_after_primary_capsule.pyfor feature visualization.

