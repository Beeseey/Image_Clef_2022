from tensorflow.keras.layers import Dense, Conv2D, LayerNormalization, MaxPooling2D, GlobalAveragePooling2D, ReLU, Dropout, Flatten
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation, Add, Reshape
import tensorflow.keras.backend as K


def my_acc(y_true,y_pred):


	y = K.abs(K.round(y_pred) - y_true)

	return K.mean(K.sum(y))

def my_loss(y_true,y_pred):

	y = K.abs(y_pred - y_true)

	return K.mean(K.sum(y))

def concept_classifier(base_model,vocab_size):


	base_model.trainable = False
	base_model_out = base_model.output
	
	base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)

	
	feature  = base_model_out

	feature = LayerNormalization()(feature)
	feature = Dense(512, activation='relu')(feature)
	feature = Dropout(0.3)(feature)
	feature = Dense(128)(feature)
	feature = LayerNormalization()(feature)
	feature = Dropout(0.5)(feature)
	feature = Flatten()(feature)
	final_model = Dense(vocab_size, activation="sigmoid")(feature)
	
	model = keras.models.Model(inputs=base_model.input, outputs=final_model)

	return model