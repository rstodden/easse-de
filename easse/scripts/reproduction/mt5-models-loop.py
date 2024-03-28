# -*- coding: utf-8 -*-


import os
import time, datetime

# !nvidia-smi

# !nvcc --version

# !pip install sentencepiece

# !pip install nvidia-ml-py3

# ! pip install transformers

# !pip install datasets

# !pip install evaluate

import json
import time
import datetime

# os.getcwd()

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from transformers.utils import logging
# logging.set_verbosity(40)

from datasets import DatasetDict, Dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, MT5Tokenizer, EarlyStoppingCallback
from datasets import load_dataset, Dataset, concatenate_datasets
import torch

import numpy as np

import evaluate

import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
gpu_name = str(nvidia_smi.nvmlDeviceGetName(handle))

print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)
print(str(nvidia_smi.nvmlDeviceGetName(handle)))

nvidia_smi.nvmlShutdown()

def read_data(data_path, file_types=["train", "dev", "test"]):
	data_name = data_path.split("/")[-2]
	if "train" in file_types:
		train_data = pd.read_csv (data_path+ data_name+"_train.csv")
		# train_data = train_data.sample(frac=1, random_state=77).reset_index(drop=True)
	else:
		train_data = None
	if "dev" in file_types:
		dev_data = pd.read_csv (data_path+ data_name+ "_dev.csv")
		# dev_data = dev_data.sample(frac=1, random_state=77).reset_index(drop=True)
	else:
		dev_data = None
	if "test" in file_types:
		test_data = pd.read_csv (data_path+ data_name+ "_test.csv")
		# test_data = test_data.sample(frac=1, random_state=77).reset_index(drop=True)
	else:
		test_data = None
	if len(file_types) == 1 and "none" in file_types:
		train_data = pd.read_csv(data_path)
	return train_data, dev_data, test_data


def read_parallel_data(data_path, file_types=["train", "dev", "test"]):
	data_name = data_path.split("/")[-2]
	if "train" in file_types:
		train_org = pd.read_csv(data_path+ data_name+"_train.org", sep="\t\t\t", header=None)
		train_org.rename(columns={0: "original"}, inplace=True)
		train_org["id"] = train_org.index
		train_simp = pd.read_csv(data_path+ data_name+"_train.simp", sep="\t\t\t", header=None)
		train_simp.rename(columns={0: "simplification"}, inplace=True)
		train_simp["id"] = train_simp.index
		train_data = pd.merge(train_org, train_simp, on=["id"])
		# train_data = train_data.sample(frac=1, random_state=77).reset_index(drop=True)
	else:
		train_data = None
	if "dev" in file_types:
		dev_org = pd.read_csv(data_path+ data_name+"_dev.org", sep="\t\t\t", header=None)
		dev_org.rename(columns={0: "original"}, inplace=True)
		dev_org["id"] = dev_org.index
		dev_simp = pd.read_csv(data_path+ data_name+"_dev.simp", sep="\t\t\t", header=None)
		dev_simp.rename(columns={0: "simplification"}, inplace=True)
		dev_simp["id"] = dev_simp.index
		dev_data = pd.merge(dev_org, dev_simp, on=["id"])
	else:
		dev_data = None
	if "test" in file_types:
		test_org = pd.read_csv(data_path+ data_name+"_test.org", sep="\t\t\t", header=None)
		test_org.rename(columns={0: "original"}, inplace=True)
		test_org["id"] = test_org.index
		test_simp = pd.read_csv(data_path+ data_name+"_test.simp", sep="\t\t\t", header=None)
		test_simp.rename(columns={0: "simplification"}, inplace=True)
		test_simp["id"] = test_simp.index
		test_data = pd.merge(test_org, test_simp, on=["id"])
	else:
		test_data = None
	if len(file_types) == 1 and "none" in file_types:
		train_data = pd.read_csv(data_path)
	# print(test_org)
	# print(test_simp)
	# print(test_data)
	return train_data, dev_data, test_data


def dataframe_to_datasetdict(train_data, dev_data, test_data):
	dataset = DatasetDict()
	if train_data is not None:
		train_dataset = Dataset.from_pandas(train_data[["original", "simplification"]])
		dataset['train'] = train_dataset
	if dev_data is not None:
		dev_dataset = Dataset.from_pandas(dev_data[["original", "simplification"]])
		dataset['dev'] = dev_dataset
	if test_data is not None:
		test_dataset = Dataset.from_pandas(test_data[["original", "simplification"]])
		dataset['test'] = test_dataset

	return dataset

def preprocess_function(dataset, data_split, tokenizer, params):
	task_prefix = params["task_prefix"]
	if "m2m" in params["model_name"]:
		tokenizer.src_lang = "de"
		tokenizer.tgt_lang = "de"
	# for seq in dataset[data_split]["original"]:
	# 	print(task_prefix, seq)
	dict_tokenizer = tokenizer(text=[task_prefix+seq for seq in dataset[data_split]["original"]], 
								text_target=dataset[data_split]["simplification"],
								max_length=params["max_input_length"], truncation=params["truncation"], 
								padding=params["padding"])	# , padding="max_length")

	dataset[data_split] = dataset[data_split].add_column("input_ids", dict_tokenizer["input_ids"])
	dataset[data_split] = dataset[data_split].add_column("attention_mask", dict_tokenizer["attention_mask"])
	dataset[data_split] = dataset[data_split].add_column("labels", dict_tokenizer["labels"])

	return dataset[data_split]

def postprocess_text(tokenizer, preds, labels, inputs=None):

	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	# print(type(inputs))
	if inputs is None:
		decoded_inputs = None
	else:
		decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

		post_processed_decoded_inputs = list()
	decoded_labels = [[reference_text] for reference_text in decoded_labels]

	return decoded_preds, decoded_labels, decoded_inputs

def compute_metric_sari(eval_preds):
				preds, labels, inputs = eval_preds
				if isinstance(preds, tuple):
						preds = preds[0]

				# Replace -100 (padding token) in the labels and inputs as we can't decode them.
				labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
				inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)

				decoded_preds, decoded_labels, decoded_inputs = postprocess_text(tokenizer, preds, labels, inputs)

				sari_result = metric_sari.compute(sources=decoded_inputs, predictions=decoded_preds, references=decoded_labels)
				return sari_result

def dataframe_to_tokenized_dataset(model_name, params, train_data=None, dev_data=None, test_data=None):
	# tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer = MT5Tokenizer.from_pretrained(model_name)
	# tokenize data
	dataset = dataframe_to_datasetdict(train_data, dev_data, test_data)
	if train_data is not None:
		tokenized_train = preprocess_function(dataset, "train", tokenizer, params)
	else:
		tokenized_train = None
	if dev_data is not None:
		tokenized_dev = preprocess_function(dataset, "dev", tokenizer, params)
	else:
		tokenized_dev = None
	if test_data is not None:
		tokenized_test = preprocess_function(dataset, "test", tokenizer, params)
	else:
		tokenized_test = None
	return tokenized_train, tokenized_dev, tokenized_test


def preprocess_data(data_path, file_types=["train", "dev", "test"], params={}, from_parallel=False):
	if from_parallel:
		train_data, dev_data, test_data = read_parallel_data(data_path, file_types)
	else:
		train_data, dev_data, test_data = read_data(data_path, file_types)
	tokenized_train, tokenized_dev, tokenized_test = dataframe_to_tokenized_dataset(params["model_name"], params, train_data, dev_data, test_data)
	return tokenized_train, tokenized_dev, tokenized_test


def build_trainer(params, tokenizer, train_tok, test_tok):
	model = AutoModelForSeq2SeqLM.from_pretrained(params["model_name"]).to(TO_DEVICE)
	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id,
		pad_to_multiple_of=64,)
	training_args = Seq2SeqTrainingArguments(output_dir=params["output_dir"],
						 learning_rate=params["learning_rate"], 
						 per_device_train_batch_size=params["batch_size"],
						 per_device_eval_batch_size=params["batch_size"],
						 weight_decay=params["weight_decay"],
						 save_total_limit=params["save_total_limit"], 
						 # num_train_epochs=params["num_train_epochs"],
						 predict_with_generate=params["predict_with_generate"],
						 fp16=params["fp16"], 
						 lr_scheduler_type = params["lr_scheduler_type"],
						 optim = params["optim"],
						 warmup_steps = params["warmup_steps"],
						 gradient_accumulation_steps = params["gradient_accumulation_steps"],
						 # per_device_train_batch_size=36,
						 # per_device_eval_batch_size=36
						 include_inputs_for_metrics=params["include_inputs_for_metrics"],
						 load_best_model_at_end=params["load_best_model_at_end"],
						 evaluation_strategy=params["evaluation_strategy"], 
						 save_strategy=params["save_strategy"],
						 # max_steps = params["max_steps"], 
						 # eval_steps = params["eval_steps"],
						 # save_steps = params["save_steps"],
						 num_train_epochs = params["num_train_epochs"],
						 # logging_steps = params["logging_steps"],
						)

	if params["early_stopping"]:
		callbacks = [EarlyStoppingCallback(3, 0.0)]
	else:
		callbacks = None
		
	trainer = Seq2SeqTrainer(
			model=model,
			args=training_args,
			train_dataset=train_tok,
			eval_dataset=test_tok,
			data_collator=data_collator,
			tokenizer=tokenizer,
			# compute_metrics=compute_metrics_rouge,
			compute_metrics=compute_metric_sari,	# compute_metrics_sari,
			callbacks=callbacks
	)
	return trainer

def train_save_model(trainer, params):
	trainer.train()

	trainer.save_model(params["save_model_path"])
	return trainer


def save_predictions(trainer, dataset, output_path, params):
	predictions, label_ids, metrics = trainer.predict(dataset, max_length=params["max_target_length"])
	decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
	output = pd.DataFrame(columns=["original", "simplification", "prediction"])
	output["original"] = dataset["original"]
	output["simplification"] = dataset["simplification"]
	output["prediction"] = decoded_preds
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
	output.to_csv(output_path+"/pred_"+timestamp+".csv", index=False)
	return output



def get_response(original, params, tokenizer, model):
	input_text = params["task_prefix"]+original
	features = tokenizer([input_text], return_tensors='pt')

	output = model.generate(input_ids=features['input_ids'],	# .to(device), 
							 attention_mask=features['attention_mask'],	# .to(device),
							 max_length=params["max_target_length"])

	return tokenizer.decode(output[0], skip_special_tokens=True)



metric_sari = evaluate.load("sari")

TO_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(TO_DEVICE)

params = {"max_input_length": 128,	# 128, 64, 512
			"max_target_length": 128,	# 128, 64, 512
			"truncation": True,
			"padding": "max_length",
			# when truncation true, padding max length and max length int given, than both truncation and padding to specific length.
			"output_dir": "models/test_trainer", 
			"fp16": False,	# True,
			"learning_rate": 0.001,	# 2e-5, 
			"batch_size": 4, # 16
			"per_device_train_batch_size": 4,	# 16,
			"per_device_eval_batch_size": 4,	# 16,
			"weight_decay": 0.01,
			"save_total_limit": 3,
			"predict_with_generate": True,
			"compute_metric": "SARI",
			# fp16=True,
			"load_best_model_at_end": True,
			# per_device_train_batch_size=36,
			# per_device_eval_batch_size=36
			"include_inputs_for_metrics": True,
			"task_prefix": "simplify to plain German: ",
			"lr_scheduler_type": "linear",
			"optim": "adafactor",
			"warmup_steps": 90,
			"gradient_accumulation_steps": 16,

			"num_train_epochs": 10,
			# "eval_steps": 5,
			# "save_steps": 5, 
			# "logging_steps": 5, 
			# "max_steps": 10,
			"evaluation_strategy": "epoch",	# "epoch",	steps
			"save_strategy": "epoch", # "epoch", steps
			"early_stopping": False,	# True,
			# "num_train_epochs": 10,	# 10,
		 }



# please download the data first and move it to the named directory.

for model_name in ["google/mt5-base"]:
	params["model_name"] = model_name
	if "own_models" in model_name:
		params["fine-tuned_model"] = "own"
		if "only_simple" in model_name:
			params["fine-tuned_model"] += "+only_simple_adaptation"
		if "cwi" in model_name:
			params["fine-tuned_model"] += "+cwi"
		if "text_leveling" in model_name:
			params["fine-tuned_model"] += "+text_leveling"
	else:
		params["fine-tuned_model"] = "none"

	# tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
	tokenizer = MT5Tokenizer.from_pretrained(params["model_name"])

	# general data
	train_tok_apa, dev_tok_apa, test_tok_apa = preprocess_data("../../resources/data/training_sets/DEplain-APA/", ["train", "dev", "test"], params)
	train_tok_web, dev_tok_web, test_tok_web = preprocess_data("../../resources/data/training_sets/DEplain-web/", ["train", "dev", "test"], params)
	train_tok_sgc, dev_tok_sgc, test_tok_sgc = preprocess_data("../../resources/data/training_sets/simple-german-corpus/", ["train", "dev", "test"], params)

	train_tok_apa_web = concatenate_datasets([train_tok_web, train_tok_apa], split="train")
	dev_tok_apa_web = concatenate_datasets([dev_tok_web, dev_tok_apa], split="dev")

	train_tok_apa_sgc = concatenate_datasets([train_tok_sgc, train_tok_apa], split="train")
	dev_tok_apa_sgc = concatenate_datasets([dev_tok_sgc, dev_tok_apa], split="dev")
	print(train_tok_apa)
	
	# test data 
	_, _, test_tok_zest = preprocess_data("../../resources/data/test_sets/sentence_level/geolino/", ["test"], params, from_parallel=True)
	_, _, test_tok_tcde19 = preprocess_data("../../resources/data/test_sets/sentence_level/TextComplexityDE/", ["test"], params, from_parallel=True)
	_, _, test_tok_bisect = preprocess_data("../../resources/data/test_sets/sentence_level/BiSECT/", ["test"], params, from_parallel=True)
	# _, _, test_tok_abgb = preprocess_data("../../resources/data/test_sets/sentence_level/ABGB/", ["test"], params, from_parallel=True)
	_, _, test_tok_apa_lha_a2 = preprocess_data("../../resources/data/test_sets/sentence_level/APA-LHA-or-a2/", ["test"], params, from_parallel=True)
	_, _, test_tok_apa_lha_b1 = preprocess_data("../../resources/data/test_sets/sentence_level/APA-LHA-or-b1/", ["test"], params, from_parallel=True)

	
	for run_i in range(7,8):
		for (train_data, dev_data), name_data in zip([(train_tok_apa, dev_tok_apa), (train_tok_apa_web, dev_tok_apa_web), (train_tok_sgc, dev_tok_sgc), (train_tok_apa_sgc, dev_tok_apa_sgc)], ["DEplain-APA", "DEplain-APA+web", "DEplain-SGC", "DEplain-APA+SGC"]):

			params["train_data"] = name_data
			params["GPU memory"] = info.total
			params["GPU name"] = gpu_name
			
			train_data = train_data.shuffle(seed=77)
			dev_data = dev_data.shuffle(seed=77)
			params["training_size"] = len(train_data)
			print(name_data, model_name)
			params["start_time"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
			modelname = params["model_name"].split("/")[-1]
			params["save_model_path"] = "models/"+modelname+"_"+name_data+"/run_"+str(run_i)
			params["output_dir"] =	"models/test_trainer-"+name_data+"-"+modelname+"/run_"+str(run_i)
			trainer = build_trainer(params, tokenizer, train_data, dev_data)
			#if ("models/" in params["model_name"] and params["fine-tuned_model"] != "none") or not "models/" in params["model_name"]:
			trainer = train_save_model(trainer, params)

			params["end_time"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

			for (dev_tok_data, test_tok_data, data_name) in [[None, test_tok_apa, "DEplain-APA"], [None, test_tok_web, "DEplain-web"], [None, test_tok_bisect, "BiSECT"], [None, test_tok_zest, "zest"], [None, test_tok_tcde19, "TextComplexityDE"], [None, test_tok_sgc, "simple-german-corpus"], [None, test_tok_apa_lha_a2, "APA_LHA_or-a2"], [None, test_tok_apa_lha_b1, "APA_LHA_or-b1"]]:
				print("test", data_name)
				if dev_tok_data is not None:
					save_predictions(trainer, dev_tok_data, "results/"+modelname+"_"+name_data+"/run_"+str(run_i)+"/result_"+data_name+"_dev", params) 
				save_predictions(trainer, test_tok_data, "results/"+modelname+"_"+name_data+"/run_"+str(run_i)+"/result_"+data_name+"_test", params)
			with open("results/"+modelname+"_"+name_data+"/run_"+str(run_i)+"/settings.json", 'w') as f:
				json.dump(params, f)
 
