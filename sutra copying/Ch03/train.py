import sys
sys.path.append('..')
# 前処理, コンテキスト・ターゲット作成, one-hotラベル変換
from common.util import preprocess, create_contexts_target, convert_one_hot
from common.trainer import Trainer # パラメータ作成?, パラメータ更新, 記録
from common.optimizer import Adam # パラメータ更新手法のAdam
from simple_cbow import SimpleCBOW # 順伝播・逆伝播の処理

window_size = 1 # 左右1つずつのコンテキスト
hidden_size = 5 # 隠れ層のサイズで入力7→5に圧縮している
batch_size = 3 # 1度に3問学習
max_epoch = 1000 # 最初から最後まで1000回行う

# 前処理でニューラルネットワークへの入力データの準備
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text) # テキスト情報を渡してIDに変換

vocab_size = len(word_to_id) # one-hotに長さをベクトルの数(配列7)を設定
contexts, target = create_contexts_target(corpus, window_size) # コーパス(ID)とウィンドウサイズを渡してコンテキストとターゲット作成
target = convert_one_hot(target, vocab_size) # targetの位置の情報とベクトルの長さを渡してone-hotにする

# ニューラルネットワーク, パラメータ更新手法, 繰り返し回数
model = SimpleCBOW(vocab_size, hidden_size) # テキストの長さ(7)と隠れ層(5)の(7, 5)を渡してニューラルネットワークを構成
optimizer = Adam() # パラメータ更新手法のAdamを使う
trainer = Trainer(model, optimizer) # CBOWでパラメータ出力させてAdamで更新する


