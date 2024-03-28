import os
import pandas as pd
import argparse
import sqlite3


def save_data_parallel(complex_texts, simple_texts, level, name, data_split, dirname, tokenized=False):
	if tokenized:
		tok_value = "tok."
	else:
		tok_value = ""
	print(len(complex_texts), len(simple_texts))
	if not os.path.exists(dirname+level):
		os.makedirs(dirname+level)
	if not os.path.exists(dirname + level + name + "/"):
		os.makedirs(dirname + level + name + "/")
	with open(dirname + level + name + "/" + name + "_" + data_split + tok_value + ".src.txt", "w", encoding="utf-8") as f:
		f.write('\n'.join(complex_texts))
	with open(dirname + level + name + "/" + name + "_" + data_split + tok_value + ".tgt.txt", "w", encoding="utf-8") as f:
		f.write('\n'.join(simple_texts))
	return 1


def clean_data(text):
	text = text.strip()
	text = text.replace("<eop>", " ")
	text = text.replace("\n", " ")
	text = text.replace("\r\n", " ")
	text = text.replace("\r", " ")
	text = text.replace(" ==== ", " ")
	text = text.replace(" === ", " ")
	text = text.replace(" == ", " ")
	text = text.replace("   ", " ")
	text = text.replace("  ", " ")
	return text.strip()


# # TextComplexityDE
def process_textcomplexityde(data_path="raw_data/parallel_corpus.csv", output_path="easse/resources/data/test_sets/"):
	# Sentence_Id,Article_ID,Article,Original_Sentence,Simplification,Rating
	textcomplexityde = pd.read_csv(data_path, header=0, encoding="ISO-8859-1")
	test_complex_clean, test_simple_clean = list(), list()

	for i, row in textcomplexityde.iterrows():
		simple_sent = row["Simplification"].strip()
		simple_sent = simple_sent.replace("\n", "")
		simple_sent = simple_sent.replace("\r", "")
		simple_sent = simple_sent.replace("\r\n", "")
		complex_sent = row["Original_Sentence"].strip()
		complex_sent = complex_sent.replace("\n", "")

		test_complex_clean.append(complex_sent)
		test_simple_clean.append(simple_sent)
	save_data_parallel(test_complex_clean, test_simple_clean, "sentence_level/", "TextComplexityDE", "test", "easse/resources/data/test_sets/")

	return 1


def process_klexikon(data_split, data_path="raw_data/klexikon/data/splits/", output_path="easse/resources/data/test_sets/"):
	klexikon_titles = os.listdir(data_path + data_split + "/klexikon")
	wiki_titles = os.listdir(data_path + data_split + "/wiki")
	parallel_files = sorted(list(set(klexikon_titles).intersection(set(wiki_titles))))
	all_complex_sentences, all_simple_sentences = list(), list()
	for i, file_path in enumerate(parallel_files):

		with open(data_path + data_split + "/wiki/" + file_path) as f:
			complex_text = (" ".join([sent for sent in f.readlines() if not sent.startswith("=")]))
			complex_text = clean_data(complex_text)
			all_complex_sentences.append(complex_text)
		with open(data_path + data_split + "/klexikon/" + file_path) as f:
			simple_text = (" ".join([sent for sent in f.readlines() if not sent.startswith("=")]))
			simple_text = clean_data(simple_text)
			all_simple_sentences.append(simple_text)
	save_data_parallel(all_complex_sentences, all_simple_sentences, "document_level/", "klexikon", data_split, output_path)
	return 1


# 20Minuten
def process_20minuten(data_split, data_path="raw_data/20Minuten/data/2021_EMNLP_newsum/EMNLP_newsum_2021_A_New_Dataset_TS_DE/2021_ANewDatasetandEfficientBaselinesforDocument-levelTextSimplificationinGerman/data/dedup/",
					  output_path="easse/resources/data/test_sets/"):
	with open(data_path + data_split + ".src.no_tag.de") as f:
		complex_docs = f.readlines()
	with open(data_path + data_split + ".trg.no_tag.simpde") as f:
		simple_docs = f.readlines()
	save_data_parallel(complex_docs, simple_docs, "document_level/", "20Minuten", data_split, output_path)
	return 1


def preprocess_apa_lha(data_path, version, split_type, simple_level, output_path="raw_data/RANLP2021-German-ATS/data/splits"):
	with open(data_path + "/" + version + "/" + split_type + "." + version + ".or") as f:
		complex_lines = f.readlines()
	with open(data_path + "/" + version + "/" + split_type + "." + version + "." + simple_level) as f:
		simple_lines = f.readlines()
	simple_lines = [line.strip() for line in simple_lines]
	complex_lines = [line.strip() for line in complex_lines]
	save_data_parallel(complex_lines, simple_lines, "sentence_level/", "APA_LHA" + version, split_type, output_path, tokenized=True)
	return 1


def preprocess_hda(data_path, output_path="easse/resources/data/test_sets/"):
	simple_texts, complex_texts = list(), list()
	conn = sqlite3.connect(data_path)
	c = conn.cursor()
	c.execute("SELECT name FROM sqlite_master WHERE type='table';")
	table_names = [t[0] for t in c.fetchall() if (t[0] != "test" and t[0] != "data")]
	for table_name in table_names:
		query = "SELECT leicht, standard FROM {}".format(table_name)
		result = c.execute(query)
		for (simple_text, complex_text) in result:
			complex_texts.append(clean_data(complex_text.strip()))
			simple_texts.append(clean_data(simple_text.strip()))
	save_data_parallel(complex_texts, simple_texts, "document_level/", "hda_easy_to_read_langauge", "test", output_path)
	return 1


parser = argparse.ArgumentParser(description='')
parser.add_argument('-test_set_name', type=str,
					help='name of the test set to be preprocessed', required=True)
parser.add_argument('-data_path', type=str, help="path of the saved data", required=True)
parser.add_argument("-output_path", type=str, help="path where to save data", required=False, default="easse/resources/data/test_sets/")

if __name__ == "__main__":
	# Read arguments from command line
	args = parser.parse_args()
	if args.test_set_name == "TextComplexityDE":
		process_textcomplexityde(args.data_path, args.output_path)
	elif args.test_set_name == "klexikon":
		# process_klexikon("test", args.data_path, args.output_path)
		process_klexikon("validation", args.data_path, args.output_path)
	elif args.test_set_name == "20Minuten":
		process_20minuten("test", args.data_path, args.output_path)
		process_20minuten("dev", args.data_path, args.output_path)
	elif args.test_set_name == "APA_LHA":
		preprocess_apa_lha(args.data_path, "or-a2", "test", "a2", output_path=args.output_path)
		preprocess_apa_lha(args.data_path, "or-b1", "test", "b1", output_path=args.output_path)
		preprocess_apa_lha(args.data_path, "or-a2", "dev", "a2", output_path=args.output_path)
		preprocess_apa_lha(args.data_path, "or-b1", "dev", "b1", output_path=args.output_path)
	elif args.test_set_name == "hda_easy-to-understand":
		preprocess_hda(args.data_path)
