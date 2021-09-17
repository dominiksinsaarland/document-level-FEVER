from transformers import BigBirdForTokenClassification, BigBirdModel, BigBirdTokenizer, BigBirdConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset

from sklearn import metrics
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import random
import json
import argparse
import re
import os
import copy

from sentence_selection_model import SentenceSelectionModel


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def process_evid(sentence):
	sentence = convert_to_unicode(sentence)
	sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
	sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
	sentence = re.sub("-LRB-", "(", sentence)
	sentence = re.sub("-RRB-", ")", sentence)
	sentence = re.sub("-COLON-", ":", sentence)
	sentence = re.sub("_", " ", sentence)
	sentence = re.sub("\( *\,? *\)", "", sentence)
	sentence = re.sub("\( *[;,]", "(", sentence)
	sentence = re.sub("--", "-", sentence)
	sentence = re.sub("``", '"', sentence)
	sentence = re.sub("''", '"', sentence)
	return sentence

class FEVERDataset(Dataset):
	def __init__(self, path, tokenizer, mode="train"):
		self.path = path
		self.mode = mode
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.examples = self.load_data()

	def load_data(self):
		examples = []
		with open(self.path) as f:
			last_id = 0
			example = {}
			lengths = []
			for line in f:
				data = json.loads(line)
				length = len(" ".join(data["sentences"]).split())
				lengths.append(length)
				if length > 1500: # these are mostly verbose lists and no "real" Wiki introduction sections
					continue

				if self.mode == "train":
					if sum(data["label_list"]) == 0:
						# drop more of the longer examples because this essentially is waste of compute
						if length > 750:
							if random.randint(0,9) != 0:
								continue
						else:
							if random.randint(0,3) != 0:
								continue

				examples.append(data)
		if self.mode == "train":
			random.shuffle(examples)		
		return examples

	def get_labels(self):
		return [0, 1]
	def get_dummy_label(self):
		return 0
	def __len__(self):
		return len(self.examples)
	def __getitem__(self, item):
		sample = self.examples[item]
		return sample
	def collate_fn(self, batch):
		all_input_ids, all_labels = [], []
		to_cap = 5
		for data in batch:
			if "conditioned_evidence" in data:
				if data["conditioned_evidence"]:
					# encode claim + first evidence sentence
					encoded_sequence = self.tokenizer.encode(data["claim"])[:-1] + self.tokenizer.encode(process_evid(data["evidence_conditioned_on"]))[1:]
				else:
					encoded_sequence = self.tokenizer.encode(data["claim"])
			else:
				encoded_sequence = self.tokenizer.encode(data["claim"])

			labels = [0] * len(encoded_sequence)
			for sent,label in zip(data["sentences"], data["label_list"]):
				sent = process_evid(sent)
				encoded_sentence = self.tokenizer.encode(sent)[1:-1] + [1]
				encoded_sequence += encoded_sentence
				labels += [label] * len(encoded_sentence)
			encoded_sequence.append(self.tokenizer.sep_token_id)
			labels.append(0)

			# CLS token should be 1 if we have at least one evidence sentence
			if sum(labels) > 0:
				labels[0] = 1

			all_input_ids.append(encoded_sequence)
			all_labels.append(labels)

		attention_masks = []
		max_length = max(len(i) for i in all_input_ids)
		for i,j in zip(all_input_ids, all_labels):
			length = len(i)
			pad_length = max_length - len(i)
			i += [self.tokenizer.pad_token_id] * pad_length
			j += [0] * pad_length
			attention_mask = [1] * length + [0] * pad_length
			attention_masks.append(attention_mask)
		input_ids = torch.tensor(all_input_ids).to(self.device)
		attention_mask = torch.tensor(attention_masks).to(self.device)
		labels = torch.tensor(all_labels).to(self.device)
		assert input_ids.shape == attention_mask.shape == labels.shape


		model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
		if self.mode == "train":
			index = (input_ids == 66).nonzero(as_tuple=True)[-1]
			index = torch.tensor([i for j,i in enumerate(index) if j % 2 == 0])
			model_inputs["sep_token_indices"] = index.to(self.device)
			return {"model_input": model_inputs, "labels": labels}
		elif self.mode == "validation":
			pages = [i["page"] for i in batch]
			sent_ids = [i["sentence_IDS"] for i in batch]
			claim_ids = [i["id"] for i in batch]
			return {"model_input": model_inputs, "labels": labels, "pages": pages, "sent_ids": sent_ids, "claim_id": claim_ids}

