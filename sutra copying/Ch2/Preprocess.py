import sys
from util import preprocess
import numpy as np

text = 'You say goodbye and I sya hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)

print(id_to_word)

def create_contexts_target(corpus, window_size=1):
    # 1つ目(window_size)と2つ目(window_size+1)の間の(-1+1)の0の位置をターゲット
    target = corpus[window_size : -window_size + 1]
    contexrs = []

    # 1～6(7-1) 先頭と末尾の文字は除外して処理する
    for idx in range(window_size, len(corpus)-window_size):
        cs = []

        # range(A, B)のBは1つ手前まで
        #  -1（1つ左）、0（ターゲット単語）、1（1つ右）となる
        for t in range(-window_size, window_size + 1):

            # 順番に入ってきたtの値が0(ターゲット)ならcontenueでループに戻る
            if t == 0:
                continue

            # idx(現在地)プラスt(左右の単語の位置)をcsリストに追加する
            cs.append(corpus[idx + t])

        # コンテキストの配列[0, 2](単語の位置)をリストで格納
        contexts.append(cs)

    # 数学的な計算ができるデータ形式に変換
    return np.array(contexts), np.array(target)

# コンテキストを抽出
contexts, target = create_contexts_target(corpus, window_size=1)

print(contexts)

print(target)

