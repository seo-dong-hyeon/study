from MyMorpheme import Morpheme
from MySentiment import Sentiment
from MyAnalyzer import Analyzer
import json

def main():
    read_file = 'ratings_test.txt'
    json_file = 'train_docs.json'
    model_name = 'tmp_model.h5'

    morpheme = Morpheme
    train_data = morpheme.read_data(morpheme, read_file)
    morpheme.write_data(morpheme, json_file, train_data)

    model = Sentiment
    model.open_file(model, json_file)
    model.make_model(model, model_name)

    analyzer = Analyzer
    analyzer.__init__(analyzer, json_file, model_name)
    analyzer.predict_pos_neg(analyzer, "올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
    analyzer.predict_pos_neg(analyzer, "배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.")
    analyzer.predict_pos_neg(analyzer, "주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ")
    analyzer.predict_pos_neg(analyzer, "믿고 보는 감독이지만 이번에는 아니네요")
    analyzer.predict_pos_neg(analyzer, "주연배우 때문에 봤어요")

if __name__ == "__main__":
	main()