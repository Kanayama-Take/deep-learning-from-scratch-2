import numpy as np

class Sigmoid:
    # 初期設定でパラメータを格納する空のインスタンス変数を準備
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x)) # 対数の数式

class Affine:
    def __init__(self, w, b):
        self.params = [w, b]

    def forward(self, x):
        W, b = self.params # 内積できるようにパラメータを別々の変数に分けて出す
        out = np.dot(x, W) + b # 内積（行列の積）＋バイアス
        return out

class TwoLayerNet:
    # 各層のニューロン数を設定する
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # パラメータサイズとランダムに数値を割り当てる
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイヤの生成
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        # TwoLayerNet内でパラメータ設定
        self.params = []
        for layer in self.layers:
            self.params += layer.params # 空リストにリストを格納

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x) # SigmoidとAffineのforwardメソッドにxを引数で渡す
        return x

# 実際の数値を指定
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict
print(s)


