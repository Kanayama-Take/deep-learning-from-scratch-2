text = 'You say goodbye and I say hello.'

# 小文字に変換する（大文字が混ざると別の単語として認識される）
text = text.lower()

# 文章にカンマやピリオドが入るように
text = text.replace('.', ' .')
print(text)

# 空白を区切り文字にして単語毎にリスタを作成
words = text.split()
print(words)

# 単語IDディクショナリ
word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id: # もしword_to_idの中にwordの単語がなければ
        new_id = len(word_to_id) # word_to_idの中の値を確認して連番を振る
        word_to_id[word] = new_id # word:ID
        id_to_word[new_id] = word # ID:word

print(id_to_word)
print(word_to_id)

# 検索
print(id_to_word[1])
print(word_to_id['hello'])

# 内包表記
import numpy as np

corpus = [word_to_id[w] for w in words] # リストにforループで値をループで追加
corpus = np.array(corpus)
print(corpus)
