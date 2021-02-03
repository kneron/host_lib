import keras
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
import numpy as np

def top_indexes(preds, n):
    sort_preds = np.sort(preds,1)
    sort_preds = np.flip(sort_preds)
    sort_index = np.argsort(preds,1)
    sort_index = np.flip(sort_index)

    for i in range(0, n):
        print(sort_index[0][i], sort_preds[0][i])
        
    return


#change the model path in your env
model = keras.models.load_model('MobileNetV2.h5')

img_path = './data/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)


x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

#output top 3 prediction
print(img_path)
top_indexes(preds, 3)

img_path = './data/fox.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

#output top 3 prediction
print(img_path)
top_indexes(preds, 3)
