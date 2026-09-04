import sys
sys.path.append('..')
from common.optimizer import SGD
from dataset import spiral
from common.trainer import Trainer
from two_layer_net import TwoLayerNet


# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
lerning_rate = 1.0

# データ読み込み、モデルのオプティマイザの生成
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr = lerning_rate)

# 
trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()