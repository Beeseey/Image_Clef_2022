import os
import csv
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import load_cids
from tensorflow.keras.models import load_model, Model
from model import get_model
import tensorflow as tf
from utils import create_tokenizer, parse_data, get_vectorizer
from generators import generator


#data_path = "../../ImageCLEF2022"
#image_path = data_path + "/e229cc37-d0da-4356-bd5c-f119c63dfacc_ImageCLEFmedCaption_2022_valid_images/valid"
#excel_path = data_path + "/46bff9d5-95d4-4362-be98-ef59819ec3af_ImageCLEFmedCaption_2022_concept_detection_valid.csv"




class History(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs = {}):
		self.train_losses = []
		self.train_acc = []
		self.val_losses = []
		self.val_acc = []

	def on_epoch_end(self, epoch, logs={}):
		self.train_losses.append(logs.get('loss'))
		self.train_acc.append(logs.get('my_acc'))
		self.val_losses.append(logs.get('val_loss'))
		self.val_acc.append(logs.get('val_my_acc'))




def get_class_weights(tokenizer, bar, caption_nums, caption_nums_dict, class_weight=dict()):

	import statistics

	low_bar = float(bar[0])
	high_bar = float(bar[1])

	std = statistics.stdev(caption_nums)
	mean = statistics.mean(caption_nums)

	for k in tokenizer.index_word:
		v = tokenizer.index_word[k].upper()
		x = abs(caption_nums_dict[v] - mean)/std
		x = 1/x
		if x < low_bar:
			class_weight[k-1] = low_bar
		elif x > high_bar:
			class_weight[k-1] = high_bar
		else:
			class_weight[k-1] = x

	return class_weight

def get_caption_list_numbers(caption_nums_dict):

	captions = []
	caption_nums = []

	for c in caption_nums_dict:

		captions.append(c)
		caption_nums.append(caption_nums_dict[c])

	return(captions,caption_nums)


def train_run(filename, image_path, batchsize, epochs, val_splitsize, bar, mode, modeloutput, data_type):

	from tensorflow.keras.applications import efficientnet

	model_path = "models"
	base_model = efficientnet.EfficientNetB0(
        			input_shape=(224,224, 3), include_top=False, weights="imagenet",
    				)

	print('------------- READING TRAINCSV FILE')
	f = open(filename, mode ='r')
	csvFile = csv.reader(f)
	parsed_data = parse_data(csvFile,float(val_splitsize), data_type)

	caption_nums_dict,train_data_dict, test_data_dict, max_ = parsed_data
	vocab_size = len(caption_nums_dict)
	

	if data_type == "concepts":
		
		captions, caption_nums = get_caption_list_numbers(caption_nums_dict)

		print('------------- CREATING TEXT TOKENIZER')
		tokenizer = create_tokenizer(captions)
	
		print('------------- CREATING CLASS WEIGHTS')
		class_weight = get_class_weights(tokenizer, bar, caption_nums, caption_nums_dict)

	elif data_type == "captions" or data_type == "concepts_seq":
		if data_type == "concepts_seq":
			data_type = "captions"
		caption_nums_dict, train_data_dict, test_data_dict, max_ = parsed_data

		vocab_size = len(caption_nums_dict)
		captions_glosary = [train_data_dict[id_] for id_ in train_data_dict]

		print('------------- CREATING TEXT VECTORIZER')
		vectorizer = get_vectorizer(captions_glosary,vocab_size, max_)

		tokenizer = vectorizer

	print('------------- CREATING DATA GENERATORS')
	train_gen = generator(mode = data_type, data_dict = train_data_dict, batch_Size = batchsize,tokenizer = tokenizer,vocab_size = vocab_size, image_path = image_path, max_length = max_)
	val_gen = generator(mode = data_type, data_dict = test_data_dict, batch_Size = batchsize,tokenizer = tokenizer,vocab_size = vocab_size, image_path = image_path, max_length = max_)


	print('------------- TRAINING MODEL')

	if not os.path.exists(model_path):
		os.mkdir(model_path)

	counter = 1

	modeloutput_ = modeloutput

	while os.path.exists(model_path+'/'+modeloutput_):

		modeloutput_ = modeloutput + "_" + str(counter)
		counter +=1

	if mode == "transformer":
		save_weights_only = True
		model_path = model_path+'/'+modeloutput_
		os.mkdir(model_path)
	else:
		save_weights_only = False

	checkpoint = ModelCheckpoint(model_path+'/'+modeloutput_, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=save_weights_only)
	history = History()

	model = get_model(base_model,vocab_size,mode,len(train_data_dict),epochs,seq_len=max_)

	
	'''
	model.fit(train_gen, epochs=epochs,
			verbose=1, steps_per_epoch=len(train_data_dict)//batchsize ,
			validation_data = val_gen, 
			validation_steps=len(test_data_dict)//batchsize,callbacks=[history,checkpoint], class_weight = class_weight)
	'''
	model.fit(train_gen, epochs=epochs,
			verbose=1, steps_per_epoch=len(train_data_dict)//batchsize ,
			validation_data = val_gen, 
			validation_steps=len(test_data_dict)//batchsize,callbacks=[history,checkpoint])

	print('------------- SAVING CONFIGS')
	
	config = ""
	config = config + " RUN FOR "+ modeloutput +"\n"
	config = config + "Model Name:\t\t"+ mode + "\n"
	config = config + "Mode:\t\t"+ data_type + "\n"
	config = config + "Batch Size:\t\t"+ str(batchsize) + "\n"
	config = config + "Split Size:\t\t" + str(val_splitsize) + "\n"
	config = config + "Class weights limits:\t\t" + str(bar) + "\n"
	config = config + "Training Losses:\t\t" + str(history.train_losses) + "\n"
	config = config + "Validatation Losses:\t\t" + str(history.val_losses) + "\n"
	config = config + "Training Accuracy:\t\t" + str(history.train_acc) + "\n"
	config = config + "Validation Accuracy:\t\t" + str(history.val_acc) + "\n"

	config = config + "\n\t\t **********************************\n\t\t**********************************\n\n"
	
	result_file_object = open('Results.txt','a')

	result_file_object.write(config)
	
	result_file_object.close()
	f.close()

	print('------------- TRAINING COMPLETE!!!')
