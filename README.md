# CO-STAR
Code and Data for the arXiv paper "CO-STAR: Conceptualisation of Stereotypes for Analysis and Reasoning" available at https://arxiv.org/abs/2112.00819

## Installation
Run the following command to install the necessary libraries:

`pip install -r requirements.txt`

## Quick Start
Pre-trained models can be downloaded from the following links:

CO-STAR-cs: https://drive.google.com/drive/folders/1kjHzz7ONWZECYI6oR2AAY4lLsRXRGBoz?usp=sharing

CO-STAR-sc: https://drive.google.com/drive/folders/1cVzROnsNmv5N6CBF4ApqJA5t_3qIitqS?usp=sharing

CO-STAR-s: https://drive.google.com/drive/folders/1nr3MJCsV_YPiLNRyP27UHXeJp14wjaZk?usp=sharing

SBF-GPT2: https://drive.google.com/drive/folders/1cp6NH6fGwNnIomF9iI8nMfrFJq7Md-kT?usp=sharing

For each model, download the entire folder and place the folder in your working directory. One of these three CO-STAR models can then be run and compared with the SBF-GPT2 model by running the following:

`python demo.py`

Make sure to change the `CS_MODEL_PATH` and `SBF_MODEL_PATH` variables in `demo.py` to their corresponding directories (and make any hyperparameter adjustments) before running the models. If you are not sure which CO-STAR model to run, try the CO-STAR-sc model first. Further details are available in the paper above.

## Training and Data Files
The CO-STAR models have been trained with the data file `./data/CO-STAR.trn.csv` which has been annotated as explained in the paper above. The models can then be evaluated with development and test data sets under `./data/SBIC.v2/SBIC.v2.dev.csv` and `./data/SBIC.v2/SBIC.v2.tst.csv` respectively.

The SBF-GPT2 model has been trained based on the paper "Social Bias Frames: Reasoning about Social and Power Implications of Language" by Sap et al. which is available at https://aclanthology.org/2020.acl-main.486/. The SBIC.v2 data set has also been annotated and provided by Sap et al. and full details can be found in the paper.

The models can be trained by running the following:

`python train_gen.py`
