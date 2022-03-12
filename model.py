import tensorflow as tf
from concept_classifier import my_acc as concept_acc
from concept_classifier import concept_classifier
from transformer import transformer_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam




class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, transformer_model, post_warmup_learning_rate, warmup_steps):
		super().__init__()
		self.transformer_model = transformer_model
		self.post_warmup_learning_rate = post_warmup_learning_rate
		self.warmup_steps = warmup_steps

	def __call__(self, step):
		global_step = tf.cast(step, tf.float32)
		warmup_steps = tf.cast(self.warmup_steps, tf.float32)
		warmup_progress = global_step / warmup_steps
		warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
		return tf.cond(
			global_step < warmup_steps,
			lambda: warmup_learning_rate,
			lambda: self.post_warmup_learning_rate,
		)

	def get_config(self):
		config = {
				'transformer_model': self.transformer_model,
				'post_warmup_learning_rate': self.post_warmup_learning_rate,
				'warmup_steps': self.warmup_steps,
				}
		return config


def get_model(base_model,vocab_size,model, train_len, epochs, seq_len=None):
	metrics = "already_defined"
	if model == "concept_classifier": 
		model = concept_classifier(base_model,vocab_size)
		loss = "binary_crossentropy"
		metrics = concept_acc
	elif model == "transformer": 
		model = transformer_model(base_model, seq_len, vocab_size)
		loss = keras.losses.SparseCategoricalCrossentropy(
					from_logits=False, reduction="none"
					)

	num_train_steps = train_len * epochs
	num_warmup_steps = num_train_steps // 15
	lr_schedule = LRSchedule(transformer_model=model, post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

	if metrics != "already_defined":
		model.compile(loss=loss, optimizer=Adam(lr_schedule), metrics=metrics)
	else:
		model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=loss)

	
	return model