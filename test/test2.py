import konlpy
import matplotlib
from konlpy.tag import Okt
import json
import os
from pprint import pprint
import nltk
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from keras.models import load_model

with open('train_docs.json', encoding='UTF8') as json_file:
    train_docs = json.load(json_file)
tokens = [t for d in train_docs for t in d[0]]

text = nltk.Text(tokens, name='NMSC')
print(text)

# 전체 토큰의 개수
print(len(text.tokens))
# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))
# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))

'''
font_fname = '/Library/Fonts/AppleGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
plt.figure(figsize=(20,10))
text.plot(50)
'''

selected_words = [f[0] for f in text.vocab().most_common(10000)]
def term_frequency(doc):
    print(doc)
    return [doc.count(word) for word in selected_words]
train_x = [term_frequency(d) for d, _ in train_docs]
train_y = [c for _, c in train_docs]
print("분석 끝")

x_train = np.asarray(train_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')
print("전처리 끝")

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_train, y_train)
print(results)
model.save('myModel.h5')

