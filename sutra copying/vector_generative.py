import numpy as np

x = np.ndarray([1, 2, 3]) # 多次元配列に対応している箱にいれる

# 入力
x.__class__ # データの方を確認
# 出力: <class 'numpy.ndarray'>

x.shape # 行列のサイズ
# 出力: (3,)

x.ndim # データの次元数を把握
# 出力: 1

# 重み（※2次元配列にするため、外側にカッコ [] を追加して修正しています）
W = np.array([[1, 2, 3], [4, 5, 6]])

W.shape # 行列のサイズ
# 出力: (2, 3)

W.ndim # データの次元数を把握
# 出力: 2