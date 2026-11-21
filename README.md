ゼロから作る Deep Learning ❷
==========================

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-2/images/deep-learning-from-scratch-2.png" width="200px">](https://www.oreilly.co.jp/books/9784873118369/)

書籍『[ゼロから作るDeep Learning ❷ ―自然言語処理編](https://www.oreilly.co.jp/books/9784873118369/)』(オライリー・ジャパン)のサポートサイトです。本書籍で使用するソースコードがまとめられています。



## ファイル構成

|フォルダ名 |説明                         |
|:--        |:--                          |
|ch01       |1章で使用するソースコード    |
|ch02       |2章で使用するソースコード    |
|...        |...                          |
|ch08       |8章で使用するソースコード    |
|common     |共通で使用するソースコード   |
|dataset    |データセット用のソースコード | 

学習済みの重みファイル（6章、7章で使用）は下記URLから入手可能です。
<https://www.oreilly.co.jp/pub/9784873118369/BetterRnnlm.pkl>

ソースコードの解説は、本書籍をご覧ください。


## Pythonと外部ライブラリ
ソースコードを実行するには、下記のソフトウェアが必要です。

* Python 3.x（バージョン3系）
* NumPy
* Matplotlib
 
また、オプションとして下記のライブラリを使用します。

* SciPy（オプション）
* CuPy（オプション）

## 実行方法

各章のフォルダへ移動して、Pythonコマンドを実行します。

```
$ cd ch01
$ python train.py

$ cd ../ch05
$ python train_custom_loop.py
```

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。

## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch-2/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>)までお知らせください。

#### この個人リポジトリでのAIとの学習

2026-09-25に追加した個人の学習方針です。上の書籍・著者教材の説明とは別に、このリポジトリで学習を再開する際に使います。

- [AIへの指示](AGENTS.md)
- [説明方法の正本](docs/TEACHING_GUIDE.md)
- [現在位置と再開メモ](LEARNING_STATE.md)
- [質問・補足説明の記録](docs/LEARNING_QUESTIONS.md)

説明は「目的 → 短いコード → 各部分の意味 → 結果の確認 → 小さな練習」の順。初登場の構文・数式・AI用語や配列のshapeは、具体例で一つずつ説明します。

#### 別PC・別AIで再開するには

初めてのPCではこのリポジトリをcloneし、既にある場合はローカル変更を確認して最新のmasterを取り込んでください。学習用Python環境は各PCで用意します。

リポジトリを開いて、AIへ次のように伝えてください。

> このリポジトリのAGENTS.md、docs/TEACHING_GUIDE.md、LEARNING_STATE.mdを最初に読み、記録された説明方法で学習を再開してください。入力するファイルと編集方法を先に示し、初めての構文・数式・AI用語を一つずつ説明してください。再開場所が分からなければ、今取り組んでいるファイルを確認してください。

リポジトリを直接読めないAIでは、上の3ファイルの内容を渡してください。SQL学習のリポジトリを一緒に取得する必要はありません。
