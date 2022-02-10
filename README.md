# Evidence Selection as a Token-Level Prediction Task

This repo contains the code and models for the paper [(Stammbach, 2021)](https://aclanthology.org/2021.fever-1.2.pdf)

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
* [sentence-selection-bigbird-base](https://www.dropbox.com/s/ui19gcjcuxfstrg/sentence-selection-bigbird-base.zip?dl=0)
* [sentence-selection-bigbird-large](https://www.dropbox.com/s/931bm5bveou1zn6/sentence-selection-bigbird-large.zip?dl=0)

* [RTE-debertav2-MNLI](https://www.dropbox.com/s/6aoi0nac2e45csi/RTE-model.zip?dl=0)

## Run the models on sample data

```shell
python src/main.py --do_predict --model_name sentence-selection-bigbird-base --eval_file sample_data.jsonl --predict_filename predictions_sentence_retrieval.csv
```



sample_data.jsonl points to a file where each line is an example of a (claim, Wiki-page) pair
* id # the claim ID
* claim # the claim
* page # the page title
* sentences # a list -- essentally the "lines" in the official FEVER wiki-pages for a given document (where the document is split by "\n")
* label_list # a list, 1 if a sentence is part of any annotated evidence set for a given claim, 0 otherwise
* sentence_IDS # a list, np.arange(len(sentences))

output is a dataframe where we store for each sentence predicted by the model
* claim_id
* page_sentence # a tuple (Wikipage_Title, sentence_ID), for example ('2014_San_Francisco_49ers_season', 3)
* y # 1 if label_list above was 1, 0 otherwise
* predictions # token-level predictions for this sentence
* score # np.mean(predictions), model is confident that this sentence is evidence if score > 0


## re-train the models

point to train_file and eval_file, both in the format described above, and add do_train flag
```shell
python src/main.py --do_train --do_predict --model_name sentence-selection-bigbird-base --eval_file sample_data.jsonl --train_file sample_data.jsonl --predict_filename predictions_sentence_retrieval.csv
```

## The pipeline
* takes a first pass over all (claim, WikiPage) pairs where Wikipages are predicted by [(Hanselowski et al., 2018)](https://github.com/UKPLab/fever-2018-team-athene) and the [FEVER baseline](https://github.com/awslabs/fever)
* extracts all sentences it is confident that they are evidence in that pass, model_input is [CLS] claim [SEP] WikiPage [SEP]
* retrieves *conditioned evidence* as explained in [(Stammbach and Neumann, 2019)](https://aclanthology.org/D19-6616/)
* retrieves hyperlinks from evidence_sentences and takes a second pass over all (claim, hyperlink) pairs where model_input is [CLS] claim, evidence_sentence [SEP] HyperlinkPage [SEP]
* sorts all predicted evidence sentences for a claim in descending order
* takes the five highest scoring sentences for each claim and concatenates those
* predicts a label for each (claim, retrieved_evidence) pair using the RTE model (trained with an outdated huggingface sequence classification demo script)

## questions

If anything should not work or is unclear, please don't hesitate to contact the authors

* Dominik Stammbach (dominik.stammbach@gess.ethz.ch)
