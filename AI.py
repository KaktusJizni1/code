import os
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = 'dataset/'

CATEGORIES = ['cats', 'dogs']

IMAGE_SIZE = (128, 128)

model = tf.keras.models.load_model("image_classifier.h5")

class_names = sorted(os.listdir(DATASET_PATH))

if len(class_names) == 2:
    class_mode = "binary"
else:
    class_mode = "categorical"


def choose_random_image():
    chosen_folder = random.choice(CATEGORIES)
    folder_path = os.path.join(DATASET_PATH, chosen_folder)

    images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png', '.jpeg'))]

    if not images:
        print("❌ Нет изображений в папке:", folder_path)
        return None

    random_image = random.choice(images)
    return os.path.join(folder_path, random_image)


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMAGE_SIZE)

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path):
    if not image_path:
        return

    try:
        img = preprocess_image(image_path)

        prediction = model.predict(img)[0]

        if class_mode == "binary":
            class_index = int(prediction > 0.5)
            confidence = prediction if class_index == 1 else 1 - prediction
        else:
            class_index = int(np.argmax(prediction))
            confidence = prediction[class_index]

        predicted_class = class_names[class_index]

        print("✅ Модель думает, что это:", predicted_class)
        print("📊 Уверенность:", round(float(confidence), 2))

        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"{predicted_class} ({round(float(confidence), 2)})")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("❌ Ошибка при предсказании:", e)


random_image_path = choose_random_image()
print("🔍 Выбрали изображение:", random_image_path)
predict_image(random_image_path)
