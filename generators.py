from utils import load_image,OHE
import os
from os.path import normpath as Path
import numpy as np
import tensorflow as tf
def concept_generator(data_dict, batch_Size,tokenizer, vocab_size,image_path):

	#IMAGES_PATH = 'C:/Users/mdrah/Downloads/Code_CLEF2018/CaptionTraining2018small'
	#print(data_dict)
	X1, y = list(), list()
	count = 0

	while True:
		
		for id_ in data_dict:

			path = os.path.join(Path(image_path),id_+'.jpg')
			
			if os.path.exists(path):
				y_ = []

				caption = data_dict[id_]
				new_label = tokenizer.texts_to_sequences([caption])[0]

				new_label = OHE(new_label,vocab_size)

				X = load_image(path,(224,224))
				
					
				X1.append(X)
				y.append(new_label)
				count+=1
			
			if count == batch_Size:
				count = 0
				X1 = np.array(X1)
				y = np.array(y)

				yield X1, y

				X1, y = list(), list()

def caption_generator(data_dict, batch_Size, tokenizer, vocab_size,image_path):

	X1, y = list(), list()
	count = 0

	while True:
		
		for id_ in data_dict:

			path = os.path.join(Path(image_path),id_+'.jpg')
			
			if os.path.exists(path):

				caption = data_dict[id_]

				x_img = load_image(path,(224,224))
				new_label = tokenizer(caption)
				X1.append(x_img)
				y.append(new_label.numpy())

				count+=1
			
			if count == batch_Size:
				count = 0
				X1 = np.array(X1)

				y = np.array(y)
				yield X1, y

				X1, y = list(), list()


def generator(mode, data_dict, batch_Size,tokenizer, vocab_size,image_path, max_length):

	if mode == "concepts":

		return concept_generator(data_dict, batch_Size,tokenizer, vocab_size,image_path)

	elif mode == "captions":

		return caption_generator(data_dict, batch_Size,tokenizer, vocab_size,image_path)