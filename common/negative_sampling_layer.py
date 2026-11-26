import sys
sys.path.append('..')
from common.np import *
from common.layers import Embedding, SigmoidWithLoss
import collections

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W) # Embeddingの引数を重みにして変数に
        self.params = self.embed.params # 重みのパラメータ
        self.grads = self.embed.grads # 勾配のパラメータ
        self.cache = None # 順伝播のときに、抜き出した重みを一時保存する

    # idx=(0, 3, 1) 単語ID
    # target_W=(3, 3)  抜き出した重み
    # W = (7, 3)  入力層 → 中間層
    # h = (3, 3) 二つの単語ベクトルを2で割った数
    def forward(self, h, idx):
        target_W = self.embed.forward(idx) # ここのidxは重み行の位置情報
        out = np.sum(target_W * h, axis=1) # 行方向に重み×単語ベクトルして合計

        self.cache = (h, target_W) # 逆伝播用に保持
        return out

    def backward(self, dout):
        h, target_W = self.cache # 順伝播のときの情報

        # shape(形状把握)でoutが(5, 122, 86)の1次元配列をreshapeで（3, 1）に転置
        # dout = 勾配
        dout = dout.reshape(dout.shape[0], 1) 

        dtarget_W = dout * h # hは(3, 3)と転置した（3, 1）で掛けて勾配を求める
        self.embed.backward(dtarget_W) # 勾配を特定（idxの位置のembedの重み）の重みに加算
        dh = dout * target_W # 勾配×重みをして入力側のEmbeddingの両方へ分かれて伝播する