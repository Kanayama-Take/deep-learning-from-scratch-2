import numpy as np
import matplotlib.pyplot as plt
from util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVDで次元削減
U, S, V = np.linalg.svd(W)

print(C[0])
print(W[0])
print(U[0])

for word, word_id in word_to_id.items(): # テキスト:単語IDを取得(.items)
    # 単語(word) SVDで削減して上位2つの座標(例：3.409e-01, -1.110e-16)を取得(重要度が高いものから自動で先頭から並んでいる)
    plt.annotate(word,(U[word_id, 0],U[word_id, 1])) # 注釈(annotate)を(単語：座標)で並べる

    # x軸の行(:,0)すべてのx座標のリストを取得(scaatter)してプロットを配置
    plt.scatter((U[:,0], U[:,1]), alpha = 0.5) 

    plt.show()


