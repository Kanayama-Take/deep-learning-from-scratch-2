import numpy as np

a = np.random.randn(3)
a.dtype # データの型を確認

# ランダムな少数が3つ入った配列を32bitに変換(as.type)
b = np.random.randn(3).astype(np.float32)
b.dtype

# astypeで引数をfでも32bitにできるショートカット
c = np.random.randn(3).astype('f')
c.dtype