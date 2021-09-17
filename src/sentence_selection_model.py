from transformers import BigBirdForTokenClassification, BigBirdModel, BigBirdTokenizer, BigBirdConfig, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
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



class SentenceSelectionModel(RobertaModel):
	def __init__(self, model_name, config, device):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.bigbird = BigBirdModel.from_pretrained(model_name, config=config, cache_dir="/cluster/work/lawecon/Work/dominik/transformer_models")
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
		self.num_labels = config.num_labels

		if os.path.exists(os.path.join(model_name, "classification layer")):
			self.classifier.load_state_dict(torch.load(os.path.join(model_name, "classification layer"),  map_location=torch.device(device)))

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		sep_token_indices=None,
	    ):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		outputs = self.bigbird(
		    input_ids,
		    attention_mask=attention_mask,
		    token_type_ids=token_type_ids,
		    position_ids=position_ids,
		    head_mask=head_mask,
		    inputs_embeds=inputs_embeds,
		    output_attentions=output_attentions,
		    output_hidden_states=output_hidden_states,
		    return_dict=return_dict,
		)
		# last hidden state

		sequence_output = outputs[0]
		
		# add dropout
		sequence_output = self.dropout(sequence_output)

		# get token level predictions
		logits = self.classifier(sequence_output)

		loss = None
		if labels is not None:
			if self.num_labels == 1:
				loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
			else:
				loss_fct = torch.nn.CrossEntropyLoss()
			# loss for each token
			loss = loss_fct(logits.view(-1), labels.view(-1).float())

			for i in range(len(attention_mask)):
				j = sep_token_indices[i]
				attention_mask[i][1:j] = False
				# mask everything before sep token (that is the claim), but not the CLS token
				# don't mask CLS token

			# only loss from relevant evidence tokens
			loss = loss * attention_mask.view(-1).float()
			# average of non-masked loss
			loss = loss.sum() / attention_mask.sum().float()

		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return TokenClassifierOutput(
		    loss=loss,
		    logits=logits,
		    hidden_states=outputs.hidden_states,
		    attentions=outputs.attentions,
		)

	def save(self, output_path, tokenizer):
		self.bigbird.save_pretrained(output_path)
		tokenizer.save_pretrained(output_path)
		torch.save(self.classifier.state_dict(), os.path.join(output_path, "classification layer"))

