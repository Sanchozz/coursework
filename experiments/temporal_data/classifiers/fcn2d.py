import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class Classifier_FCN2D:

	def __init__(self, input_shape, nb_classes, verbose=True, build=True, lr=0.0001):
		self.output_directory = 'out_fcn2d'
		if build == True:
			self.model = self.build_model(input_shape, nb_classes, lr=lr)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
		return

	def build_model(self, input_shape, nb_classes, lr=0.0001):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv2D(64, kernel_size=(3, 3),padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling2D()(conv3)

		output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['AUC'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=lr)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val, batch_size=16, epochs=2000, weights={0 : 1., 1: 1.}):
		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))
        
		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks, class_weight=weights)

		self.model.save(self.output_directory+'last_model.hdf5')
		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		keras.backend.clear_session()

	def predict(self, x_test,return_df_metrics=False):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred