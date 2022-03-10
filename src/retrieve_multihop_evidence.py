import argparse
from fever_doc_db import FeverDocDB
import pandas as pd
from collections import defaultdict
from ast import literal_eval
import json
import unicodedata
import re

def get_predictions(df):
	preds = defaultdict(list)
	print (df)
	for score, page_sent, idx in zip(df["predictions"], df["page_sentence"], df["claim_id"]):
		preds[idx].append((score,list(page_sent)))
	return preds

def fetch_documents(db, pages):
	docs = defaultdict(lambda: [])
	for page, lines in db.get_all_doc_lines(pages):
		docs[page] = re.split("\n(?=\d+)", lines)
	return docs

def get_hyperlinks_from_sent(docs, page, sent_id):
	hyperlinks = set()
	for i, hyperlink in enumerate(docs[page][sent_id].split("\t")[2:]):
		if i % 2 == 1:
			item = normalize_text_to_title(hyperlink)
			hyperlinks.add(item)
	return hyperlinks

def normalize(text):
	return unicodedata.normalize('NFD', text)

def normalize_text_to_title(text):
	return normalize(text.strip()\
		.replace("(","-LRB-")\
		.replace(")","-RRB-")\
		.replace(" ","_")\
		.replace(":","-COLON-")\
		.replace("[","-LSB-")\
		.replace("]","-RSB-"))


def sanity_check(sent):
	# some sentences are empty
	if not sent:
		return False
	# some are too short
	if len(sent) <= 3:
		return False
	# some are from disambiguation pages and thus not valuable
	if "may refer to" in sent:
		return False
	return True

def get_sentences(doc):
	for sent in doc:
		if sent:
			yield sent.split("\t")[1]
		else:
			yield ""

def get_full_evidence_set(example):
	# return set of all (page, sent_ID) in evidence_set (so that we do not randomly sample from one of these for tricky cases)
	evidence_set = set()
	if example["verifiable"] == "NOT VERIFIABLE":
		return evidence_set
	evidence = [[(unicodedata.normalize("NFD", i[2]) ,i[3]) for i in x] for x in example["evidence"]]
	for evid in evidence:
		evidence_set.update(evid)
	return evidence_set



def get_example(db, example, hyperlinks):
	docs = fetch_documents(db, hyperlinks)
	res = []

	evidence_set = get_full_evidence_set(example)
	for page, sents in docs.items():
		out = {"id": str(example["id"])}
		sent_list, label_list, sent_ids = [], [], []
		for sent_ID, sent in enumerate(get_sentences(sents)):
			#print (sent)
			if sanity_check(sent):
				sent_list.append(sent)
				if (page, sent_ID) in evidence_set:
					label_list.append(1)
				else:
					label_list.append(0)
				sent_ids.append(sent_ID)
		out["sentences"] = sent_list
		out["label_list"] = label_list
		out["claim"] = example["claim"]
		out["sentence_IDS"] = sent_ids
		out["page"] = page
		res.append(out)
	return res

def load_examples(filename):
	examples = {}
	with open(filename) as f:
		for i in f:
			i = json.loads(i)
			examples[i["id"]] = i
	return examples

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--db_file", default="fever.db", type=str, help="path to fever db")
	parser.add_argument("--predictions", default="predictions_sentence_retrieval_sample.csv", type=str, help="path to output file of predicting evidence in a first pass")
	parser.add_argument("--fever_data", default="../fever-transformers/data/fever/dataset/dev.jsonl", type=str, help="path to train/dev/test split of FEVER data")
	parser.add_argument("--outfile_name", default="multihop_evidence_sample_data.jsonl", type=str, "outfile name for the multihop prediction pass")

	args = parser.parse_args()

	# initalize db and examples
	db = FeverDocDB(args.db_file)
	fever_examples = load_examples(args.fever_data)
	df = pd.read_csv(args.predictions)

	# in multihop, we're just interested in evidence which we predicted before
	df = df[df.score >= 0]
	df.page_sentence = df.page_sentence.apply(lambda x: literal_eval(x))
	predictions = get_predictions(df)

	covered = defaultdict(set)
	with open(args.outfile_name, "w") as outfile:
		for claim_id, predicted_pages in predictions.items():
			predicted_pages = sorted(predicted_pages, key=lambda x:x[0], reverse=True)
			for score, (page, sent_ID) in predicted_pages:
				docs = fetch_documents(db, [page])
				# sentence
				sent = docs[page][sent_ID].split("\t")[1]
				hyperlinks = get_hyperlinks_from_sent(docs, page, sent_ID)

				hyperlinks = [i for i in hyperlinks if i not in covered[claim_id]]
				covered[claim_id].update(hyperlinks)

				example = fever_examples[claim_id]
				out = get_example(db, example, hyperlinks)

				for i in out:
					i["conditioned_evidence"] = True
					i["evidence_conditioned_on"] = sent
					i["orig_page"] = page
					json.dump(i, outfile)
					outfile.write("\n")



