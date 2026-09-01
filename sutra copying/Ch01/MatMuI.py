import numpy as np

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W = self.params
        out = np.dot(x, W)
        self.x = x
        return

    def backward(self, dout):
        W = self.params
        dx = np.dot(dout, W.T) # 入力の勾配 dx は、損失の情報を持つ dout と、重みの転置 W.T を掛け合わせる（内積）ことで計算されます。
        dw = np.dot(self.x.T, dout) # 重みの勾配 dw は、入力の転置 x.T と、損失の情報を持つ dout を掛け合わせる（内積）ことで計算されます。
        self.grads[0][...] = dw 
        return dx
