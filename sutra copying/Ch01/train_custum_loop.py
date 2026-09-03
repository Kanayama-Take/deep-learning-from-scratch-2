import sys
sys.path.append('..')
from common.optimizer import SGD
from dataset import spiral
from two_layer_net import TwoLayerNet
import numpy as np
import matplotlib.pyplot as plt

# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
lerning_rate = 1.0

# データ読み込み、モデルのオプティマイザの生成
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr = lerning_rate)

# 学習で使用する変数
data_size = len(x) # xのデータ数を把握
max_iters = data_size // batch_size # バッチサイズがデータサイズに対して何回学習すれば1通り（1エポック）になるか
total_loss = 0
loss_count = 0
loss_list = []


for epoch in range(max_epoch): # range(何回まで)
    idx = np.random.permutation(data_size) # データサイズ(300-1=299)までの値でランダム値
    x = x[idx]
    t = t[idx]

    # 「1回分のまとまり（ミニバッチ）」を順番に切り出し
    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size] # [0:30]= (0*30):(1*30)
        batch_y = t[iters*batch_size:(iters+1)*batch_size]
        
        # 勾配を求めてパラメータ更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 学習経過を出力
        if (iters+1) % 10 == 0:#30+1,2,3...と40になったら、10で割り切れる(%10)と処理される
            avg_loss = total_loss / loss_count # 合計された誤差を回数で割って平均誤差を出す
            print('| epoch %d |  iter %d / %d | loss %.2f'
                              % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss) # lossの学習曲線を描くためのリストに追加して蓄積
            total_loss, loss_count = 0, 0 # 1イテレーションごとに算出するためにリセット

