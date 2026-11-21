import numpy as np

class Embedding :
    def __init__(self, W): 
        self.params = [W] # (7, 3)行列 
        self.grads = [np.zeros_like(W)] # Wと同じ配列の勾配
        self.idx = None # 今回抜き出す行番号

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward1(self, idx):
        dW, = self.grads # 修正する重みの配列を同じ(7, 3)にする
        dW[...] = 0 # ...で配列を維持して要素だけ0にする
        dW[self.idx] = dout # 順伝播の時に保持した位置(idx)に勾配を当てる
        return None

    def backward2(self, idx):
        dw, = self.grads
        dw[...] = 0

        for i, word_id in enumerate(self, dout):
            dw[word_id] += dout[i]