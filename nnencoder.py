#We know this version is depracated
import warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
import matplotlib.pyplot as plt
##Need tensorflow 1.14 to use the contrib module
import tensorflow as tf
import logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
##Other stuff.
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.contrib import ffmpeg
from scipy.fftpack import rfft, irfft
from glob import iglob
from pydub import AudioSegment
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.data import Dataset
import pickle

##Fit to a folder of dual channel WAV files, name that folder audio_wav.
#Seemed to work best
inputSize = 12348
x = []
#So this doesn't take too long.
numbertotrain = 100
#Session acts like an environment manager
with tf.Session() as sess:
	#Create an array of arrays, each array inside being a two dimensional array of the following, for each file
		#[ [left1], [right1,], [left2], [right2]...[leftn], right[n]], each a chunk of the size inputSize, taking steps of size inputSize each time.
	def process_wav():
		file_range = 0
		#Each file gets its own two-dimensional array.
		fileIdx = 0
		for file in iglob('audio_wav' +'/*.wav'):
			#So this doesn't take too long.
			if fileIdx < numbertotrain:
				audio_binary = tf.read_file(file)
				wav_decoder = audio_ops.decode_wav(
					audio_binary,
					desired_channels=2)
				sample_rate, audio = sess.run([wav_decoder.sample_rate, wav_decoder.audio])
				fileAudio = np.array(audio)
				#Only use sounds of the same length, this length seems to match most.
				if len(fileAudio) == 5294592:
				#Audio is split into two channels, cut each one into inputSize sized chunks and store them sequentially.
				#Use discrete fourier transforms to map from the time domain into the frequency domain.
					leftAudio = rfft(audio[:,0])
					rightAudio = rfft(audio[:,1])
					#Split the both arrays into subarrays of length inputSize
					lower = 0
					upper = inputSize
					#Sliding window.
					while upper < len(leftAudio):
						leftAudioSection = leftAudio[lower:upper]
						rightAudioSection =  rightAudio[lower:upper]
						#Add them in sequential order
						x.append(leftAudioSection)
						x.append(rightAudioSection)
						lower += inputSize
						upper += inputSize
					#Now x contains the subarrays we'd like, in the order [ [left1], [right1,], [left2], [right2]...[leftn], right[n]]
				else:
					pass
				print("preprocessed file: " + str(file) + ", Number: " + str(fileIdx))
			fileIdx += 1

	#Store the the input data in x
	process_wav()
	#Type it all properly.
	x = np.array(x)

	#Model specification.
	t_model = Sequential()
	#Input layer
	t_model.add(Dense(256, name = 'encode', input_shape=(inputSize,)))
	#Bottleneck layer for dimensionality reduction.
	##Should definetly, use more hidden layers, one person used 
	#hidden_1_size = 8400
	#hidden_2_size = 3440
	#hidden_3_size = 2800

	t_model.add(Dense(128, name='bottleneck'))
	#Output layer.
	t_model.add(Dense(inputSize, activation=tf.nn.sigmoid))
	t_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				loss=tf.losses.sigmoid_cross_entropy)
	#Model training.
	numEpochs = 10 #Should definetly be above, even 1k-10k is undershooting this. Loss may not even converge to zero in this number of epochs.
	t_model.fit(x, x, batch_size = 50, epochs = numEpochs)

	#Broken, not sure why
	##Save the fit model to disk.
	#filename = 'finalized_model.sav'
	#pickle.dump(t_model, open(filename, 'wb'))
		# To load the model from disk
		# loaded_model = pickle.load(open(filename, 'rb'))
		# result = loaded_model.score(X_test, Y_test)

	#Now the model is fit. Let's do some stuff with it.
	#Get input tensor
	def get_input_tensor(model):
		return model.layers[0].input
	# get bottleneck (dimensionality reduction mapping) tensor
	def get_encode_tensor(model):
		return model.get_layer(name='encode').output
	# Get output tensor
	def get_output_tensor(model):
		return model.layers[-1].output

	#We can get all the variables we'd like through these lines:
	t_input = get_input_tensor(t_model)
	t_enc = get_encode_tensor(t_model)
	t_dec = get_output_tensor(t_model)
	# enc will store the actual encoded (hopefully compressed) values of x
	enc = sess.run(t_enc, feed_dict={t_input:x})
	# dec will store the actual decoded values of enc
	dec = sess.run(t_dec, feed_dict={t_enc:enc})

	def save_encoded_song_data(folder, encodedObj):
		with open(str(folder) + "outfile", "w") as outfile:
			outfile.write("\n".join(str(encodedObj)))

