import tensorflow as tf
from tensorflow import keras as keras

flags = tf.app.flags
flags.DEFINE_string("input", None, "training data [csv: <path time image>, <flags>]\nIf not specified, there will not be any training")
flags.DEFINE_list("tags", None, "the tags to detect")
flags.DEFINE_integer("validation", 100, "specified how many samples shall be used for validation")
flags.DEFINE_integer("samplesize", None, "how many samples the csv contains in total", lower_bound=0)
flags.DEFINE_integer("epochs", 50, "how many epochs the network should train for", lower_bound=1)
flags.DEFINE_string("model", None, "the file of the model[used to load and save the model]")
flags.DEFINE_spaceseplist("classify", [], "the paths of the images to classify")
flags.DEFINE_string("outfile", "classified.csv", "the path of the csv file that the new labels will be written to")

FLAGS = flags.FLAGS

import sys

if len(sys.argv)==1:
	print(FLAGS.get_help())
	exit(0)
else:
	FLAGS(sys.argv)

model = None

if FLAGS.model:
	try:
		model = keras.models.load_model(FLAGS.model,compile=True)
	except(ImportError,ValueError,IOError):
		print("Error during import")
		pass
if not model:
	print("Building model")
	model = keras.Sequential()
	
	model.add(keras.layers.Conv2D(
		input_shape=(None,None,3),
		kernel_size=8,
		filters=1,
		data_format="channels_last"))
	
	model.add(tf.keras.layers.MaxPool2D(pool_size=4,
		data_format="channels_last"))
	
	model.add(tf.keras.layers.Conv2D(kernel_size=25,
		filters=FLAGS.tags,
		data_format="channels_last"))
	
	model.add(tf.keras.layers.GlobalMaxPool2D(
		data_format="channels_last"))
	
	model.add(tf.keras.layers.Dense(len(FLAGS.tags),
		activation=tf.nn.softmax))

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
	loss='mse')

model.summary()

def _preprocess(imagefile):
	image=tf.image.decode_image(tf.read_file(imagefile))
	image=tf.image.convert_image_dtype(image,tf.float32)
	image=image[:,:,:3]
	image=tf.cond(tf.less(tf.shape(image)[2],tf.constant(3)),lambda: tf.image.grayscale_to_rgb(image[:,:,:1]),lambda:image)
	return image

def _parse_function(files,*tags):
	image=_preprocess(files)
	return (image,tf.stack(tags))

if FLAGS.input:
	print("Parsing input")
	dataset=tf.contrib.data.CsvDataset(FLAGS.input,[tf.string]+[tf.float32]*len(FLAGS.tags),header=True)
	dataset=dataset.map(_parse_function).batch(1)
	validation=dataset.take(FLAGS.validation)
	dataset=dataset.skip(FLAGS.validation).repeat(FLAGS.epochs)
	print("Training...")
	model.fit(dataset,epochs=FLAGS.epochs,steps_per_epoch=FLAGS.samplesize-FLAGS.validation)
	print("Done")
	
if FLAGS.model:
	model.save(FLAGS.model)

pred=tf.data.Dataset.from_tensor_slices(FLAGS.classify).map(_preprocess)

import csv
with open(FLAGS.outfile,"wb" if sys.version_info <= (2,) else "w") as outfile:
	csvout=csv.writer(outfile)
	csvout.writerow(["path"]+FLAGS.tags)
	for i in range(len(FLAGS.classify)):
		out=[FLAGS.classify[i]]
		for j in model.predict(tf.stack([_preprocess(FLAGS.classify[i])]),steps=1)[0]:
			out.append(j)
		csvout.writerow(out)
		