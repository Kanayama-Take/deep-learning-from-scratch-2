import sys 
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess # corpus(idの配列), id:word, word:idのデータを前処理で入れる
vacab_size = len(word_to_id)
C = create_co_matrix(corpus, vacab_size) # ウィンドウサイズを使って共起行列を作成

# クエリ(検索単語), 単語→ID, ID→単語, 共起行列, 上位5つの類似度
most_similar('you', word_to_id. id_to_word, C, top=5) # コサイン類似度を算出して出力