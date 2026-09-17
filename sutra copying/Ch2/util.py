import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    # (語彙数,　語彙数)の形状で0を振る、32bitで
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    # corpusの位置（何番目）と値のセットで取り出す(例 2:4)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            # ウィンドウズが左にはみ出てないときに以下の処理
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            # ウィンドウズサイズが右にはみ出ていないときに以下の処理
            if right_idx < corpus_size:
                right_word_id = corpus[left_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

# コサイン類似度
def cos_similarity(x, y, eps=1e-8):
    # 分母：単語1(x)・単語2(y)データをそれぞれ2乗和して平方根で計算してものでx・ｙを割る
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id : # query(検索単語)がword_to_idの辞書の中に無い場合
        print('%s is not found' % query) # queryをisの前に代入(%s)させる
        return # 処理終了
    
    # ベクトルを取り出す
    print('/n[query]' + query)
    query_id = word_to_id[query] # query(検索単語)からword_to_idの対応する単語のiD取得
    query_vec = word_matrix[query_id] # IDから対応行のベクトルを取得

    # コサイン類似度
    vocab_size = len(id_to_word) # 辞書・対応表内の単語数を取得
    similarity = np.zeros(vocab_size) # 単語数と同じ0だけのリストを作成

    # queryと1つずつ類似度を照らし合わせて算出
    for i in range(vocab_size):
        # 共起行列を元にqueryのベクトルとi番目のベクトルのコサイン類似度を算出してリストに追加するループ処理
        similarity[i]= cos_similarity(word_matrix[i], query_vec) 

    # 結果を出力
    count = 0
    for i in (-1 * similarity).argsort(): # argsortが昇順のみ対応なので、マイナスをかけて逆転させている
        if id_to_word[i] == query:
            continue
        print('/s: /s' % (id_to_word[i], similarity[i])) # id_to_wordの単語と類似度を表示

        count += 1
        if count >= top: # top5よりも現在のカウントが多かったら処理を止める
            return

def ppmi(C, verbose=False, eps=1e-8): # 共起行列, 状況確認のためのログ, 0にならないように極小値
    M = np.zeros_like(C, dtype=np.float32) # 共起行列と同じ形状で0で埋めて、float32で小数にも対応しておく
    N = np.sum(C) # 共起行列の数値を合計して近似
    S = np.sum(C, axis=0) # 共起行列の各単語毎の出現回数
    total = C.shape[0] * C.shape[1] # [0]で列方向と[1]行方向のサイズ(shape)を計算
    cnt = 0

    for i in (C.shape[0]): # 縦方向の数(単語の数)だけループ
        for j in (C.shape[1]): # 横方向の数だけループ

            # 単語xと単語yの共起回数(詳細はNotion) * 全体数(N) / 単語x出現数 * 単語y出現数 + 極小値 = logで算出
            pmi = np.log2(C[i, j] * N /(S[j] * S[i]) + eps) # (10 * 10000 / 1000 * 20) + eps
            M[i, j] = max(0, pmi) # 0と算出したpmiを比較して一番大きい数値(max)を格納、0の丸め誤差対策

            if verbose:
                cnt += 1

                #  # 10000 // 100 + 1 = 101 → 50(cnt) / 101=0.49 → 0でないためprintはスキップ(101だと1)
                if cnt % (total//100 + 1) == 0:

                    print('%.1f%% done' % (100*cnt/total)) # 100 * 101 / 10000 = 1.01 小数点を除外されるので1.0 doneとなる。

    return M    


