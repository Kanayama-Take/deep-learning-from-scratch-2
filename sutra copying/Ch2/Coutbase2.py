from util import preprocess

# 前処理用の関数を呼び出し
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(id_to_word)
print(word_to_id)