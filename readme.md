# Fast and Efficient Registration of Subsampled MRI Data Using Neural Networks Utilizing K-Space Properties

This repository contains all of the work/code done for my masters thesis in 2024. The work was supervised by Prof. Dr. Mattias Heinrich, M.Sc. Ziad Al-Haj Hemidi and M.Sc. Eytan Kats from the Institute for Medical Informatics of the University Lübeck.

## Structure

There are multiple folders
* Images: Contains all generated images
* ModelParameters: Contains all model parameters for ACDC and CMRxRecon respectively
* TestResults: Contains all test results for parameter tests on and domain translation between ACDC and CMRxRecon as well as for the reconstruction pipeline using the CMRxRecon data

All other files are a collection of scripts that either contain the specific dataset in their name or have parameters to configure the dataset to use. Most of the names are self-explainatory like Test_Results_ for different script that generate the test results. Wandb_ stands for [Weights and Biases](https://wandb.ai/site) which was used to better track the experiments for the parameter tests. 
Helper functions are stored in Functions.py for all other scripts. Similarly Models.py contains all model architectures used in the experiments.

## Data

For the experiments the [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and [CMRxRecon](https://cmrxrecon.github.io/Home.html) datasets were used. This data is not provided here, but is publicly available.

## Acknowledgments

This work is based on the [Fourier-Net](https://github.com/xi-jia/Fourier-Net) repository as well as using code from the [IC-Net](https://github.com/zhangjun001/ICNet) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph) projects.