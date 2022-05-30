from transformers import BigBirdForTokenClassification, BigBirdModel, BigBirdTokenizer, BigBirdConfig, AutoTokenizer
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
from sentence_selection_dataset import FEVERDataset


def train(args, train_dataset, eval_dataset):

	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, collate_fn = eval_dataset.collate_fn, batch_size=args.batch_size)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=args.batch_size)

	warmup_steps = 0
	t_total = int(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	model.zero_grad()
	use_amp = True
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	for epoch in range(args.num_epochs):
		bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, args.num_epochs)
		epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
		print('Train ...')
		for step, batch in enumerate(epoch_iterator):
			model.train()


			with torch.cuda.amp.autocast(enabled=use_amp):
				loss = model(**batch["model_input"], labels=batch["labels"])
				loss = loss[0]
				loss = loss / args.gradient_accumulation_steps
			scaler.scale(loss).backward()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()
				optimizer.zero_grad()
		model.save(args.save_path, tokenizer)
		evaluate(model, eval_dataloader, tokenizer, predict_filename)

def evaluate(model, eval_dataloader, tokenizer, predict_filename):
	model.zero_grad()
	epoch_iterator = tqdm(eval_dataloader)
	predictions, labels = [], []
	input_ids = []
	pages = []
	sent_ids = []
	claim_ids = []
	for i, batch in enumerate(epoch_iterator):
		with torch.no_grad():
			model.eval()
			# predict labels for each token in a document
			output_i = model(**batch["model_input"]).logits.squeeze()
			if len(output_i.shape) == 1:
				output_i = output_i.unsqueeze(0)

			# append predictions
			predictions.append(output_i.cpu().tolist())
			# and labels
			labels.append(batch["labels"].cpu().tolist())
			# and input_ids (we need those later for splitting claim and evidence
			input_ids.append(batch["model_input"]["input_ids"].cpu().tolist())
			# add the page name to save the output
			pages.append(batch["pages"])
			# add ids of individual sentences
			sent_ids.append(batch["sent_ids"])
			# and finally the claim ID
			claim_ids.append(batch["claim_id"])

	# This could be done more efficiently
	# in a nutshell, we iterate over all predictions
	# and split by [SEP] token (everything before [SEP] is the claim, we can just ignore that
	# and then, we iterate over the sequence after the [SEP] token, until we encounter an <EOS> token
	# if we encounter <EOS>, we save predictions for that sentence (alongside the page, the sendence_ID and the claim_ID)
	y_true_agg, y_pred_agg = [], []
	out_tuples = []

	all_claim_ids = []
	for batch_input_id, batch_y_true, batch_y_pred, page, sent_id, claim_id in zip(input_ids, labels, predictions, pages, sent_ids, claim_ids):
		for input_id, y_true, y_pred, page, sent_id_, page_, claim_id_ in zip(batch_input_id, batch_y_true, batch_y_pred, pages, sent_id, page, claim_id):
			flag_first_sequence = False
			counter_sent_id = 0
			try:
				for i,j,k in zip(input_id, y_true, y_pred):
					# end of claim
					if i == tokenizer.sep_token_id and not flag_first_sequence:
						flag_first_sequence = True
						tmp, tmp_y = [], []
						continue
					# claim
					if not flag_first_sequence:
						continue
					# only padding tokens from here on
					if flag_first_sequence and i == tokenizer.sep_token_id:
						break
					if i == tokenizer.pad_token_id:
						continue
					# sentence finished here
					if i == 1:
						tmp.append(k)
						tmp_y.append(j)
						y_pred_agg.append(tmp)
						y_true_agg.append(tmp_y)
						out_tuples.append((page_, sent_id_[counter_sent_id]))
						all_claim_ids.append(claim_id_)
						counter_sent_id += 1
						tmp, tmp_y = [], []
						continue
					tmp.append(k)
					tmp_y.append(j)
			except Exception as e:
				print ("exception", str(e))
				print (input_id, y_true, y_pred)

	sentence_y, sentence_pred = [], [] # compute sentence level pr/rc/f1 -- this is not necessarily the same as in the FEVER Score which has some peculiarities
	stats_y, stats_pred = [], []
	out_dataframe = []
	for i,j,k,l in zip(y_pred_agg, y_true_agg, out_tuples, all_claim_ids):
		# we have an annotated sentence (either all labels are 0 or 1)
		if sum(j) > 0:
			sentence_y.append(1)
		else:
			sentence_y.append(0)
		if np.mean(i) > 0:
			sentence_pred.append(1)
		else:
			sentence_pred.append(0)
		stats_y.extend(j)
		stats_pred.extend(i)
		out_dataframe.append((sentence_pred[-1], i, k, l))

	df = pd.DataFrame(out_dataframe, columns =['y', 'predictions', 'page_sentence', 'claim_id'])
	df["score"] = df["predictions"].apply(lambda x: np.mean(x))
	df.to_csv(predict_filename)
	classification_report = metrics.classification_report(sentence_y, sentence_pred)
	cm = metrics.confusion_matrix(sentence_y, sentence_pred)
	print (classification_report)
	print (cm)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
		        help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--batch_size", default=2, type=int,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--num_epochs", default=2, type=int,
		        help="Total number of training epochs to perform.")
	parser.add_argument("--learning_rate", default=2e-5, type=float,
		        help="The initial learning rate for Adam.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
		        help="Epsilon for Adam optimizer.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
		        help="Weight decay if we apply some.")
	parser.add_argument("--only_eval", action="store_true",
		        help="Whether to run evaluation.")
	parser.add_argument("--get_embeddings", action="store_true",
		        help="Whether to run evaluation.")
	parser.add_argument("--i", type=str, default=None, help="")
	parser.add_argument("--use_translations", action="store_true",
		        help="Whether to run evaluation.")
	parser.add_argument("--iteration", type=str, default=None, help="")
	parser.add_argument("--entropy_weight", default=2, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--model_name", default="sentence-selection-bigbird-base", type=str,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--dropout", default=None, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--train_file", default="sample_data.jsonl", type=str,
		        help="path to train file")
	parser.add_argument("--eval_file", default="sample_data.jsonl", type=str,
		        help="path to evaluation file")
	parser.add_argument("--save_path", default="sentence-selection-bigbird", type=str,
		        help="name of save directory")
	parser.add_argument("--predict_filename", default="predictions_sentence_retrieval.csv", type=str,
		        help="path to file in which predictions are stored")

	parser.add_argument('--do_train', action='store_true')
	parser.add_argument('--do_predict', action='store_true')

	args = parser.parse_args()
	device = "cuda" if torch.cuda.is_available() else "cpu"

	config = BigBirdConfig.from_pretrained(args.model_name)
	config.gradient_checkpointing = True
	config.num_labels = 1
	model = SentenceSelectionModel(args.model_name, config, device).to(device)

	tokenizer = AutoTokenizer.from_pretrained(args.model_name)

	if args.do_train:
		print ("training")
		eval_dataset = FEVERDataset(fn_val, tokenizer, mode="validation")
		train_dataset = FEVERDataset(args.train_file, tokenizer, mode="train")
		print (len(train_dataset))
		train(args, train_dataset, eval_dataset)

	if args.do_predict:
		print ("evaluating")
		eval_dataset = FEVERDataset(args.eval_file, tokenizer, mode="validation")
		eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, collate_fn = eval_dataset.collate_fn, batch_size=args.batch_size)
		evaluate(model, eval_dataloader, tokenizer, args.predict_filename)
	print ("finished")

	# add do_train, do_predict
	# refactor evaluate
	# test both on sample data
	# done
