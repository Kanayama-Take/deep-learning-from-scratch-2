import sys
sys.path.append('..')
from layers import Affine, Sigmoid, SoftmaxWithLoss
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みの初期化
        W1 = 0.01 * np.random.randn(I, H) 
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 順伝播・逆伝播のパラメータの重み,バイアスと勾配を1つに格納しえて、一括更新しやすくしている。
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 順伝播
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            return x

    # 誤差を算出
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t) # 予測データと正解ラベル
        return loss

    # 逆伝播
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            return dout
