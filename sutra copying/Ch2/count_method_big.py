import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
# 共起行列(create_co_matrix), コサイン類似度(most_similar), 単語の関連度スコアに変換(ppmi)
from common.util import create_co_matrix, most_similar, ppmi
from dataset import ptb

window_size = 2 # 文章が長いので広めに設定
wordvec_size = 100 # 1万以上の単語数をそのまま行列の次元数だと重いので上位100を残す

# trainデータを計算可能なIDに変換
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id) # 共起行列のマス目に使う単語の数を算出

# 共起行列
print('couting co-occrrence') # 共起行列の計算を始めるとうメッセージ
C = create_co_matrix(corpus, vocab_size, window_size) # (1万, 1万)の共起行列

# PPMI
print('calculating PPMI ...')
W = ppmi(C, verbose=True) # 共起行列を真の関連度スコアの行列に変換し処理の途中経過を表示する(verbose=True)

# SVD 
print('calculating SVD ...')
try:
    from sklearn.utils.extmath import randomized_svd
    # 抽出したい重要軸(n_componets)を100(wordvec_size)指定してUに入れる
    # n_iter=5で重要軸を正確にするための近似する処理
    # 値は固定せずに乱数で行うので毎回少し違う結果になる
    U, S, V = randomized_svd(W, n_components = wordvec_size, n_iter=5,random_state=None)

except ImportError: # tryでsklernがない場合の受け皿
    U, S, V = np.linalg.svd(W) # NunpyでSVDを計算する(遅い)

# スライシング
word_vecs = U[:, :wordvec_size] # SVD計算したUを100(wordvec_size)個分、戦闘を切り取る(:, N)

# 検索テスト
querys = ['you', 'year', 'car', 'toyota']

for query in querys:
    # 100次元のベクトル空間上に対して検索した単語が最も意味的に近い上位5つを表示させる
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
