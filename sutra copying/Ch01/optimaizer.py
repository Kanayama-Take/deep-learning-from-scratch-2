import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in (len(params)):
            params[i] -= self.lr * grads[i]

model = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    x_batch, y_batch = get_mini_batch(...) # ミニバッチを取得
    loss = model.forward(x_batch, y_batch) # ミニバッチを引数に順伝播をして損失の勾配をだす
    model.backward() # 逆伝播
    optimizer.update(model.paramsm model.grads) # パラメータ更新