import yaml
import shutil
import os
import subprocess
import sys
from multiprocessing import Process
import numpy as np
from functools import partial
import pandas as pd

sys.path.append("./")


def save_json(filepath, content, append=False):
	"""
	Saves content to a JSON file

	:param filepath: path to a file (must include .json)
	:param content: dictionary of stuff to save

	"""
	if not 'json' in filepath:
		raise ValueError("filepath is invalid")

	if not append:
		with open(filepath, 'w') as json_file:
			json.dump(content, json_file, indent=4)
	else:
		with open(filepath, 'w+') as json_file:
			json.dump(content, json_file, indent=4)


def save_yaml(filepath, content, append=False, topcomment=None):
	"""
	Saves content to a yaml file

	:param filepath: path to a file (must include .yaml)
	:param content: dictionary of stuff to save

	"""
	if not 'yaml' in filepath:
		raise ValueError("filepath is invalid")

	if not append:
		method = 'w'
	else:
		method = 'w+'

	with open(filepath, method) as yaml_file:
		if topcomment is not None:
			yaml_file.write(topcomment)
		yaml.dump(content,yaml_file, default_flow_style=False, indent=4)

def load_json(filepath):
	"""
	Load a JSON file

	:param filepath: path to a file

	"""
	if not os.path.isfile(filepath) or not ".json" in filepath.lower(): raise ValueError("unrecognized file path: {}".format(filepath))
	with open(filepath) as f:
		data = json.load(f)
	return data

def load_yaml(filepath):
	"""
	Load a YAML file

	:param filepath: path to yaml file

	"""
	if filepath is None or not os.path.isfile(filepath): raise ValueError("unrecognized file path: {}".format(filepath))
	if not "yml" in filepath and not "yaml" in filepath: raise ValueError("unrecognized file path: {}".format(filepath))
	return yaml.load(open(filepath), Loader=yaml.FullLoader)


def load_feather(path):
    return pd.read_feather(path)

def save_df(df, filepath):
        df.to_pickle(filepath)

def load_df(filepath):
        return pd.read_pickle(filepath)

