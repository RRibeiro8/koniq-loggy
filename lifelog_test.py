from kuti import applications as apps
from kuti import image_utils as iu

import pandas as pd, numpy as np, os
from matplotlib import pyplot as plt
from tqdm import tqdm

from PIL import UnidentifiedImageError

model_root = 'trained_models/'
data_root = 'datasets/LSC21/'

ids = pd.read_csv(data_root + 'lsc21-images.csv')

from tensorflow.keras.models import Model

base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
head = apps.fc_layers(base_model.output, name='fc', 
                      fc_sizes      = [2048, 1024, 256, 1], 
                      dropout_rates = [0.25, 0.25, 0.5, 0], 
                      batch_norm    = 2)    

model = Model(inputs = base_model.input, outputs = head)

model.load_weights(model_root + 'b8_MSE_withkuti_best_weights.h5')#'original_koncep512-trained-model.h5')

model.summary()

print(ids.image_path)

data = []

for img_path in tqdm(ids.image_path.values):

	# Load an image
	#print(img_path)
	image_path = data_root + 'lsc21-image/' + img_path
	
	try:
		im = preprocess_fn(iu.read_image(image_path, image_size=(384,512)))
		# Create a batch, of 1 image
		batch = np.expand_dims(im, 0)

		# Predict quality score
		y_pred = model.predict(batch).squeeze()

		data.append([img_path, y_pred])
		#print(img_path)
		#print(f'Predicted score: {y_pred:.{2}f}')

	except UnidentifiedImageError:
		print("Image Corrupted!", img_path)
		

df = pd.DataFrame(data, columns=['image_path', 'MOS'])
df.to_csv('b8kuti_lsc21-MOS.csv', index = False, header = True)
print(df)