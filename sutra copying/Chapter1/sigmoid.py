import numpy as np

def sigmoid(x):
    return 1 / (1  + np.exp(-x)) # シグモイド関数の数式

x = np.random.randn(10, 2) # 入力10 出力2
W1 = np.random.randn(2, 4) # 隠れ層 入力2 出力4
b1 = np.random.randn(4) # バイアスのスカラー値
W2 = np.random.randn(4, 3) # 隠れ層
b2 = np.random.randn(3) 

h = np.dot(x, W1) + b1 # 隠れ層1
a = sigmoid(h) # シグモイド関数を通して出力
s = np.dot(a, W2) + b2 # 出力されたものを隠れ層に入力