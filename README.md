# Evidence Selection as a Token-Level Prediction Task

This repo contains the code and models for the paper [(Stammbach, 2021)](https://fever.ai/workshop.html)

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n FEVER_bigbird python=3.6
conda activate FEVER_bigbird

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Models

The models (pytorch models) can be downloaded here:
* [sentence-selection-bigbird-base](https://www.dropbox.com/s/931bm5bveou1zn6/sentence-selection-bigbird-large.zip?dl=0)
* [sentence-selection-bigbird-large](https://www.dropbox.com/s/ui19gcjcuxfstrg/sentence-selection-bigbird-base.zip?dl=0)
* [RTE-debertav2-MNLI](https://www.dropbox.com/s/6aoi0nac2e45csi/RTE-model.zip?dl=0)

## Run the models on sample data



## re-train the models

## questions

If anything should not work or is unclear, please don't hesitate to contact the authors

* Dominik Stammbach (dominik.stammbach@gess.ethz.ch)
