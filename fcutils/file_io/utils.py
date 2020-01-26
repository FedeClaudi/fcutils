import os
import json
import pandas as pd
from collections import namedtuple
import requests
import yaml


def connected_to_internet(url='http://www.google.com/', timeout=5):
	"""
		Check that there is an internet connection

		:param url: url to use for testing (Default value = 'http://www.google.com/')
		:param timeout:  timeout to wait for [in seconds] (Default value = 5)
	"""
	
	try:
		_ = requests.get(url, timeout=timeout)
		return True
	except requests.ConnectionError:
		print("No internet connection available.")
	return False



def listdir(fld):
	"""
	List the files into a folder with the coplete file path instead of the relative file path like os.listdir.

	:param fld: string, folder path

	"""
	if not os.path.isdir(fld):
		raise FileNotFoundError("Could not find directory: {}".format(fld))

	return [os.path.join(fld, f) for f in os.listdir(fld)]

def get_subdirs(folderpath):
	"""
		Returns the subfolders in a given folder
	"""
	return [ f.path for f in os.scandir(folderpath) if f.is_dir() ]


def check_create_folder(folderpath):
	# Check if a folder exists, otherwise creates it
	if not os.path.isdir(folderpath):
		os.mkdir(folderpath)

def check_folder_empty(folderpath):
	if not len(os.listdir(folderpath)):
		return True
	else:
		return False

def check_file_exists(filepath):
	# Check if a file with the given path exists already
	return os.path.isfile(filepath)
