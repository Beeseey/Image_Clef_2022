from tensorflow.keras.layers import TextVectorization
import re

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import efficientnet
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
'''

def get_cnn_model():
	base_model = efficientnet.EfficientNetB0(
		input_shape=(224,224,3), include_top=False, weights="imagenet",
	)
	# We freeze our feature extractor
	base_model.trainable = False
	base_model_out = base_model.output
	base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
	cnn_model = keras.models.Model(base_model.input, base_model_out)
	return cnn_model

def get_cnn_model2(base_model):

	base_model.trainable = False
	base_model_out = base_model.output
	
	base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
	cnn_model = keras.models.Model(base_model.input, base_model_out)
	return cnn_model

from tensorflow.keras.applications import DenseNet121

#model_path = "models"
base_model = DenseNet121(input_shape=(224,224,3),weights='imagenet')
#base_model = Model(inputs=base_model.input,outputs=)
base_model.trainable = False
base_model = keras.models.Model(inputs=base_model.input,outputs=base_model.layers[-119].output)

model = get_cnn_model2(base_model)

print(model.summary())

'''
def custom_standardization(input_string):
	strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
	strip_chars = strip_chars.replace("<", "")
	strip_chars = strip_chars.replace(">", "")
	lowercase = tf.strings.lower(input_string)
	return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

vectorization = TextVectorization(
	max_tokens=10,
	output_mode="int",
	output_sequence_length=6,
	standardize=custom_standardization,
)

data = ["The boy, has eyes","I am a girl with legs","boy is king","girl is queen","King has eyes"]

vectorization.adapt(data)

print(vectorization("The boy"))

'''

a = vectorization("boy with eyes")
b = tf.convert_to_tensor([[0.01,0.7,0.01,0.01,0.01,0.05,0.01,0.04,0.16,0.16],
							[0.01,0.7,0.01,0.01,0.01,0.05,0.01,0.04,0.16,0.16],
							[0.01,0.01,0.7,0.01,0.01,0.05,0.01,0.04,0.16,0.16],
							[0.01,0.7,0.01,0.01,0.01,0.05,0.01,0.04,0.16,0.16],
							[0.01,0.7,0.01,0.01,0.01,0.05,0.01,0.04,0.16,0.16],
							[0.01,0.7,0.01,0.01,0.01,0.05,0.01,0.04,0.16,0.16]])
print(np.argmax(b.numpy()))
cross_entropy = SparseCategoricalCrossentropy(
	from_logits=False, reduction="none"
)

def calculate_loss(loss, y_true, y_pred, mask = None):
	loss = loss(y_true, y_pred)
	mask = tf.math.not_equal(y_true, 0)
	print(mask)
	mask = tf.cast(mask, dtype=loss.dtype)
	print(loss)
	loss *= mask
	print(loss)
	print(tf.reduce_sum(mask))
	return tf.reduce_sum(loss) / tf.reduce_sum(mask)
	

print(calculate_loss(cross_entropy,a,b))
'''