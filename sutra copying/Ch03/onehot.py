import sys
sys.path.append('..')
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'

# コーパスと単語にIDを振る処理(preprocess)
corpus, word_to_id, id_to_word = preprocess(text)

# コンテキストとターゲットを2次元配列で抽出して格納
contexts, target = create_contexts_target(corpus, window_size=1)

# 単語の数
vocab_size = len(word_to_id)

# ターゲットの位置(target) , 単語の数(vocab_size)を引数にしてターゲットのiDだけ1にする。
target = convert_one_hot(target, vocab_size)

# コンテキストの位置(contexts) , 単語の数(vocab_size)を引数にしてコンテキストのiDだけ1にする。
contexts = convert_one_hot(contexts, vocab_size)