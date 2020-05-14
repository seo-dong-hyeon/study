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
import tensorflow as tf

okt = Okt()

with open('train_docs.json', encoding='UTF8') as json_file:
    train_docs = json.load(json_file)
tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    print(doc)
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

selected_words = [f[0] for f in text.vocab().most_common(10000)]
def term_frequency(doc):
    print(doc)
    return [doc.count(word) for word in selected_words]

def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))

model = tf.keras.models.load_model('myModel.h5')
predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
predict_pos_neg("배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.")
predict_pos_neg("주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ")
predict_pos_neg("믿고 보는 감독이지만 이번에는 아니네요")
predict_pos_neg("주연배우 때문에 봤어요")

