
import numpy as np

def cos_similarity(x, y, eps=1e-8):
    # 分母：単語1(x)・単語2(y)データをそれぞれ2乗和して平方根で計算してものでx・ｙを割る
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)