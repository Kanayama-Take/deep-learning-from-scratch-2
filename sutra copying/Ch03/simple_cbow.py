import sys
sys.path.append('..')
from common.layers import MatMul, SoftmaxWithLoss
import numpy as np

class SimpleCBOW:
    # 入力層・出力層（vocab_size）隠れ層の数（hidden_size）
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size 

        # 重みを入力層×隠れ層のサイズでバンダムに作成後、0.01をかけて爆発を抑える
        # 32nitにしてデータを軽くする(astype('f'))
        w_in = 0.01 * np.random.randn(V,H).astype('f')

        # 隠れ層×出力層のサイズで重みを作成して0.01をかける
        w_out = 0.01 *  np.random.randn(H,V).astype('f')

        # レイヤの生成
        # レイヤー0,1は左右のコンテキストを入れる口
        self.in_layer0 = MatMul(w_in)
        self.in_layer1 = MatMul(w_in)

        # 出力層
        self.out_layer = MatMul(w_out)

        # スコアの出口を作る、loss
        self.loss_layer = SoftmaxWithLoss

        # 全てのレイヤーの勾配とパラメータを順番に取り出す
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.grads, self.params = [], []
        for layer in layers:
            self.grads += layer.grads
            self.params += layer.params

        # 学習を終えた重みを変数に格納する。
        self.word_vec = w_in

    def forward(self, contexts, target):
        # (バッチ分全て(:),列の0番目(0)) 左のコンテキストだけを指定して入口(h0)に
        h0 = self.in_layer0(contexts[:, 0])
        h1 = self.in_layer1(contexts[:, 1]) # 反対

        # 左右のコンテキストを合算して0.5で割って平均を出すことで中間層のデータ(h)になる
        h = (h0 + h1) * 0.5

        # 統合されたhを順伝播に通して予測スコアが出力
        score = self.out_layer.forward(h)

        # 計算されたscoreとtargetを渡して、確率に変換されlossが算出される
        loss = self.loss_layer.forward(score, target)
        return loss

    # 逆伝播
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout) # 勾配1を誤差逆レイヤに渡す
        da = self.out_layer.backward(ds) # 前回の勾配(ds)を受け取って伝播
        da *= 0.5 # ×0.5のノードはそのままかけて伝播

        # 足し算の逆伝播はそのまま値を流すだけ
        self.in_layer0.backward(da) 
        self.in_layer1.backward(da)

        return None


