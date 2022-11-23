"""
author: takeru sakata
data: 2022/11/22
"""
import csv
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer as PS
nltk.download('punkt')
nltk.download('stopwords')


def main():
    """この部分は実行に時間が掛かるのでcorpusをpickleで事前にダンプした"""
    all_comment = Extract_Comment("gameDataCSV.csv")
    corpus = make_corpus(all_comment)
    f = open('corpus.bin', "wb")
    pickle.dump(corpus, f)


def Extract_Comment(file_name):
    """
    csvデータからコメントのみ抽出する
    """
    with open(file_name, "r", encoding="utf-8") as csv_file:
        f = csv.reader(csv_file)
        all_comment = [row[5] for row in f]  # コメントのみを取り出す

    return all_comment


def make_corpus(all_comment):
    """
    取得したコメントから[['word',...,'word'],... ,['word',..., 'word']]
    という形式のコーパスを作る。
    """
    corpus = []
    i = 0
    for comment in all_comment:
        # remove stop words
        text_tokens = word_tokenize(comment)
        tokens_without_sw = [
            word for word in text_tokens if not word in stopwords.words()]

        # stemming
        stemmed_tokens = []
        for word in tokens_without_sw:
            stemmed_tokens.append(PS().stem(word))

        # カンマ、ピリオドが残っているので取り除く
        stemmed_tokens = [token for token in stemmed_tokens if (
            token != '.') and (token != ',')]
        corpus.append(stemmed_tokens)
        print(i)
        i += 1
    return corpus


if __name__ == "__main__":
    main()
