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
okt = Okt()

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))
predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
predict_pos_neg("배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.")
predict_pos_neg("주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ")
predict_pos_neg("믿고 보는 감독이지만 이번에는 아니네요")
predict_pos_neg("주연배우 때문에 봤어요")
