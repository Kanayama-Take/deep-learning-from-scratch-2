import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset import ptb

# コーパスの学習データを使って、プロットに必要な前処理(ptb.load_data)をする
corpus, word_to_id, id_to_word = ptb.load_data('train')

print('corpus_size:', len(corpus)) # コーパスの単語数を算出
print('corpus[:30]:', corpus[:30]) # 先頭30個だけ切り出す(:30)
print() # 見やすくするための空行
print('id_to_word[0]', id_to_word[0]) # idが0の単語を出力
print('id_to_word[1]', id_to_word[1]) # idが1の単語を出力
print("word_to_id['car']", word_to_id['car']) # 単語のcarのidを出力
