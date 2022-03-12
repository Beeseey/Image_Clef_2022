from PIL import ImageFile,Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
import csv
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import re

def load_image(path, target_size):
	
	image = load_img(path, target_size=target_size)
	image = img_to_array(image)

	image = preprocess_input(image)

	return image

def OHE(index_list,length):

	label = np.zeros(length)

	for index in index_list:
		label[index-1] = 1
	return label

def create_tokenizer(descriptions):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(descriptions)
	return tokenizer

def my_acc(y_true,y_pred):

	y = K.abs(K.round(y_pred) - y_true)

	return K.mean(K.sum(y))

def my_loss(y_true,y_pred):

	y = K.abs(y_pred - y_true)

	return K.mean(K.sum(y))

def load_cids(csv_data, val_splitsize, cids = dict(), train_image_cid_dict = dict(), test_image_cid_dict = dict(), max_ = 0):
	
	csv_data = list(csv_data)[1:]
	n = len(csv_data)

	lstm_dict = dict()
	

	for num,line in enumerate(csv_data):
		line = line[0].split("\t")
		img_id = line[0]
		line_cids = line[1].split(';')

		if num%2 == 0 and len(test_image_cid_dict) < int(len(csv_data) * val_splitsize):
			test_image_cid_dict[img_id] = line_cids
		else:
			train_image_cid_dict[img_id] = line_cids
		for cid in line_cids:
			#print(cid)
			if cid not in cids:
				cids[cid] = 1
			else:
				cids[cid]+=1

			max_ = max(max_, len(line_cids))

	
	return (cids,train_image_cid_dict,test_image_cid_dict,max_)

def load_caption_ids(csv_data, val_splitsize, cids, train_image_cid_dict, test_image_cid_dict, max_, concepts_seq = False):
	
	csv_data = list(csv_data)[1:]
	n = len(csv_data)

	lstm_dict = dict()
	

	for num,line in enumerate(csv_data):
		line = line[0].split("\t")
		img_id = line[0]
		if not concepts_seq:
			line_cids = line[1].split(' ')
			line = line[1]
		else:
			line_cids = line[1].split(';')
			line = " ".join(line_cids)

		if num%2 == 0 and len(test_image_cid_dict) < int(len(csv_data) * val_splitsize):
			test_image_cid_dict[img_id] = "start " + line + " end"
		else:
			train_image_cid_dict[img_id] = "start " + line + " end"
		for cid in line_cids:
			#print(cid)
			if cid not in cids:
				cids[cid] = 1
			else:
				cids[cid]+=1

		max_ = max(max_, len(line_cids))
	
	return (cids,train_image_cid_dict,test_image_cid_dict,max_)


def parse_data(csv_data, val_splitsize, data_type, cids = dict(), train_image_cid_dict = dict(), test_image_cid_dict = dict(), max_ = 0):

	if data_type == "captions" or data_type == "concepts_seq":

		if data_type == "concepts_seq":

			return load_caption_ids(csv_data, val_splitsize, cids, train_image_cid_dict, test_image_cid_dict, max_,True)

		return load_caption_ids(csv_data, val_splitsize, cids, train_image_cid_dict, test_image_cid_dict, max_)

	else:

		return load_cids(csv_data, val_splitsize, cids, train_image_cid_dict, test_image_cid_dict, max_)

def custom_standardization(input_string):
	strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
	strip_chars = strip_chars.replace("<", "")
	strip_chars = strip_chars.replace(">", "")
	lowercase = tf.strings.lower(input_string)
	return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def get_vectorizer(captions,vocab_size, max_len):

	vectorize_layer = TextVectorization(
										max_tokens=vocab_size+2,
										output_mode="int",
										output_sequence_length=max_len,
										standardize=custom_standardization,
									)
	vectorize_layer.adapt(captions)

	return vectorize_layer

