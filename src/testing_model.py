import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import json

from preprocessing_tools import \
    CTCLayer, \
    encode_single_sample, \
    num_to_char, \
    decode_batch_predictions


#                   Load the data

with open('../../SingleLineClearImagesTest/words_coordinates.json') as json_file:
    DATA = json.load(json_file)


def take_clear_image_text(clear_image_name):
    res = ''
    clear_image_data = DATA[clear_image_name]

    for word_data in clear_image_data:
        word = str(word_data['word'])
        if word.find('\n') == -1:
            res += word + ' '
        else:
            res += word
    return res


# Path to the data directory
data_dir = Path('../../SingleLineDegradedImagesTest/')

# Get list of all the images
images = list(map(str, list(data_dir.glob("*.png"))))
images = sorted(images, key=lambda x: int(''.join(re.findall('\d+', x))))
labels = [take_clear_image_text('clear_image'+str(i)+'.png') for i in range(len(images))]

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))


#                   Preprocessing

# Batch size for training and validation
batch_size = 16


# Maximum length of any captcha in the dataset
max_label_len = max([len(label) for label in labels])

for i in range(len(labels)):
    labels[i] += ' ' * (max_label_len - len(labels[i]))


indices = np.arange(len(images))
np.random.shuffle(indices)
x_test, y_test = np.array(images)[indices], np.array(labels)[indices]


#                   Create Dataset objects

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = (
    test_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)


#                   Loading the model

model = load_model('./models/keras_model.h5', custom_objects={'CTCLayer': CTCLayer})


#                   Inference

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()


#  Let's check results on some validation samples
for batch in test_dataset.take(2):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds, max_label_len)

    _, ax = plt.subplots(4, 4, figsize=(10, 5))
    for i in range(16):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        label = tf.strings.reduce_join(num_to_char(batch_labels[i])).numpy().decode("utf-8")
        title = f"True:\n {label}\nPrediction:\n {pred_texts[i]}\n"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title, size=6)
        ax[i // 4, i % 4].axis("off")
plt.show()
