# Загрузка библиотек
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pprint import pprint

from easyocr import Reader
# import pytesseract 

# Задание модели Yolov8
model = YOLO("weights/best.pt")

# Задание easyOCR
reader = Reader(['ru'], gpu=True)

# Чтение изображений
filenames = os.listdir('sources')
imgs = list()
for name in filenames:
    imgs.append(cv2.imread("sources/{}".format(name)))

# Поиск номеров на изображении
results = model.predict(imgs, save=False)

# Цикл по всем изображениям
res = dict()
for i, result in enumerate(results):
    img = imgs[i]

    # Задание словаря атрибутов для изображения
    res[filenames[i]] = dict()
    res[filenames[i]]['name'] = filenames[i]
    res[filenames[i]]['numbers'] = dict()

    # Извлечение координат ограничивающих рамок номеров
    boxes = result.boxes.xyxy.cpu().numpy()

    # Цикл по всем найденным номерам
    for box in boxes:
        # Извлечение номера из изображения        
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        crop = img[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Извлечение текста из изображения
        text = reader.readtext(crop_gray)[0][1]

        # Задание списка атрибутов
        atr = list()

        # Определение принадлежности к ведомствам на основе преобладающего цвета
        b, g, r = np.mean(crop[0]), np.mean(crop[1]), np.mean(crop[2])
        # print(b, g, r)

        # Внесение информации в словарь
        res[filenames[i]]['numbers'][text] = dict()
        res[filenames[i]]['numbers'][text]['number'] = text
        res[filenames[i]]['numbers'][text]['atr'] = atr

        # Отрисовка рамок на изображении
        cv2.rectangle(img, (x1, y1), (x2, y2),
                     (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

# Вывод результата
pprint(res)
cv2.destroyAllWindows()
