import sys
sys.path.append('..')
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text) # 前処理を入れる
vocab_size = len(word_to_id) 
C = create_co_matrix(corpus, vocab_size) # ウィンドウズサイズから共起行列を作成

# cos類似度の算出
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))