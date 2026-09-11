# preprocess(前処理), create_co_matrix(共起行列), cos_similarity(コサイン類似度), ppmi(PPMI行列に変換)
from util import preprocess, create_co_matrix, cos_similarity, ppmi
import numpy as np

text = 'You say goodbye and I say hello.'
# 前処理したテキストをコーパス, ID：word, word:IDで保存
corpus, word_to_id, id_to_word = preprocess[text] 
vocab_size = len(word_to_id) #  word:IDのセットの数を数える
C = create_co_matrix(corpus, vocab_size) # 共起行列を作成
W = ppmi(C) # 単語関連度

np.set_printoptions(precision=3) # 3桁まで表示
print('co-occurrence matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)