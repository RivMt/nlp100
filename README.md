# 言語処理100本ノック 2025 — 第6章〜第9章

富士フイルム自然言語処理課題コードです。

各問題の実際の実行結果は、[GitHub Releases](https://github.com/RivMt/nlp100/releases)で
公開しています。出力内容や学習済みモデルを確認する際に参照してください。

## 実行方法

### 環境設定

> 本課題はPython 3.10.11, Windows 11で作成されました。

Pythonの仮想環境を作成し、依存パッケージをインストールします。以下は
PowerShellで実行する例です。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

問題77を実行する場合は、CUDA対応版のPyTorchが必要です。本環境のRTX 3060では、
CUDA 13.0版のPyTorchを次のようにインストールしています。

```powershell
python -m pip install --force-reinstall "torch==2.13.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

2行目で`True`とGPU名が表示されることを確認してください。CUDAのバージョンは、
使用するGPUドライバとPyTorchが対応するものを選択してください。

以降のコマンドは、プロジェクトのルートディレクトリで実行してください。

### コマンドから実行

`main.py`に実装済みの問題番号（50〜89）を渡して実行します。

```powershell
python main.py 50
python main.py 60
python main.py 70
python main.py 77
python main.py 80
python main.py 87
```

問題固有の引数は、問題番号の後ろに指定します。

```powershell
python main.py 65 "an excellent movie"
```

`main.py`は対象問題の`PROBLEM_DESCRIPTION`を表示してから、登録された
`main()`を現在のPythonプロセス内で呼び出します。

### VS Codeから実行

1. プロジェクトのルートディレクトリをVS Codeで開きます。
2. コマンドパレットの「Python: Select Interpreter」から
   `venv\Scripts\python.exe`を選択します。
3. 「実行とデバッグ」ビューで`Pydbg: main.py`を選択して開始します。
4. 引数入力欄に問題番号と必要な引数を入力します。例: `65 "an excellent movie"`

デバッグ設定は[`.vscode/launch.json`](.vscode/launch.json)にあります。環境変数を
利用する場合は、プロジェクトルートに`.env`を作成すると、この設定から読み込まれます。

## コード構成

```text
.
├── main.py              # 問題番号から実行する関数を選択するエントリーポイント
├── chap06/              # 第6章: 単語ベクトル（問題50〜59）
│   ├── 50.py〜59.py     # 各問題の実装
│   └── common.py        # 第6章で共通して利用する処理
├── chap07/              # 第7章: 機械学習（問題60〜69）
│   ├── 60.py〜69.py     # 各問題の実装
│   └── common.py        # 第7章で共通して利用する処理
├── chap08/              # 第8章: ニューラルネット（問題70〜79）
│   ├── 70.py〜79.py     # 各問題の実装
│   └── common.py        # データ変換、モデル、学習、評価の共通処理
├── chap09/              # 第9章: BERT型事前学習済みモデル（問題80〜89）
│   ├── 80.py〜89.py     # 各問題の実装
│   └── common.py        # トークン化、学習、評価の共通処理
├── utils/               # 章をまたいで利用する共通処理
├── res/                 # データセットおよびモデルのキャッシュ
└── out/                 # 問題ごとの実行結果
```

各問題は`chapXX/YY.py`に実装され、先頭に`PROBLEM_DESCRIPTION`、実行処理として
`main()`を持ちます。章内の共通処理は`chapXX/common.py`、全章共通の処理は
`utils/`に配置しています。

前の問題の結果を利用する問題では、必要な`out/<問題番号>/`のファイルを確認します。
すべて存在する場合は再実行せず、不足している場合に限り、先行問題の`main()`を
同じプロセス内で実行します。

- 問題55は、問題54が出力する`capital_common_countries.tsv`を利用します。
- 問題62は、問題61が出力する学習データの特徴ベクトルを利用し、学習済みモデルと
  `DictVectorizer`を保存します。
- 問題63〜68は、問題62が保存したモデルと`DictVectorizer`を利用します。検証データや
  評価データが必要な処理では、問題61の特徴ベクトルも利用します。
- 問題69は、問題61の学習・検証データを利用し、正則化パラメータごとにモデルを
  学習します。
- 問題70〜79は、共通の単語埋め込みキャッシュとSST-2の変換処理を利用します。
- 問題74は、問題73が保存した`model.pt`を利用します。モデルが存在しない場合は、
  問題73を先に実行します。
- 問題76〜79は、問題75で実装したパディング処理を共通の`collate`関数として
  利用します。
- 問題86、87、89は、問題85が保存したSST-2のトークン列を利用します。不足している
  場合は問題85を先に実行します。
- 問題88は、問題87が保存したファインチューニング済みモデルを利用します。

問題54の対象は`capital-common-countries`セクションのみであるため、その出力に
文法的アナロジーの事例は含まれません。

## `res`と`out`

`res/<データセット名>/`には、ダウンロードしたデータセットやモデルのキャッシュを
保存します。主なデータは次のとおりです。

- `res/questions-words/questions-words.txt`: 単語アナロジー評価データ
- `res/WordSimilarity-353/wordsim353.tsv`: WordSimilarity-353評価データ
- `res/GoogleNews-vectors-negative300.bin`: Google NewsのWord2Vecバイナリ
- `res/embeddings/<モデル名>.pt`: 第8章の単語埋め込み行列と語彙のキャッシュ
- `res/SST-2/train.tsv`: SST-2の学習データ
- `res/SST-2/dev.tsv`: SST-2の検証データ

単語ベクトルはNumPyを用いたバイナリローダーで直接読み込みます。デフォルトでは
`res/GoogleNews-vectors-negative300.bin`を使用します。以前にgensimで取得した
`res/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz`が存在する
場合は、互換性のため圧縮ファイルも読み込めます。

`out/<問題番号>/`には、各問題が生成したJSON、TSV、画像、学習済みモデルなどを
保存します。例えば、問題54の結果は`out/54/`、問題61の特徴ベクトルは`out/61/`、
問題62の学習済みモデルは`out/62/`、問題69のグラフは`out/69/`に保存されます。
第8章では、データ変換やパディングの確認結果を`out/70/`〜`out/75/`に、
学習履歴とモデルを`out/73/model.pt`および`out/76/`〜`out/79/`に保存します。
第9章では、トークン化・類似度・ミニバッチの結果を`out/80/`〜`out/86/`に、
学習履歴とモデルを`out/87/`〜`out/89/`に保存します。

`res/`と`out/`の内容はGitの管理対象外です。必要に応じて再生成してください。
実際に生成した結果は[GitHub Releases](https://github.com/RivMt/nlp100/releases)から
確認できます。

## 環境変数

環境変数はPowerShellで設定するか、VS Codeから実行する場合は`.env`に記述します。

| 環境変数 | デフォルト値 | 説明 |
| --- | --- | --- |
| `NLP100_WORD2VEC_PATH` | `res/GoogleNews-vectors-negative300.bin` | Word2Vecバイナリファイルのパス |
| `NLP100_ANALOGY_LIMIT` | 制限なし | analogyデータで各セクションから使用する最大事例数 |
| `NLP100_RESTRICT_VOCAB` | 制限なし | 類似語検索で使用する語彙数 |
| `NLP100_EMBEDDING` | `glove-wiki-gigaword-50` | 第8章で使用するgensimの単語埋め込み |
| `NLP100_EPOCHS` | `5` | 第8章の学習エポック数 |
| `NLP100_MAX_EXAMPLES` | 制限なし | SST-2の各splitから読み込む最大事例数 |
| `NLP100_BERT_MODEL` | `bert-base-uncased` | 第9章で使用するBERT型モデル |
| `NLP100_MAX_LENGTH` | `128` | 第9章で使用する最大トークン列長 |
| `NLP100_BATCH_SIZE` | `16` | 第9章のファインチューニング時のバッチサイズ |

例:

```powershell
$env:NLP100_WORD2VEC_PATH = "res/GoogleNews-vectors-negative300.bin"
$env:NLP100_ANALOGY_LIMIT = "10"
$env:NLP100_RESTRICT_VOCAB = "50000"
python main.py 54
```

第8章を短時間で動作確認する場合は、使用する事例数とエポック数を制限できます。

```powershell
$env:NLP100_MAX_EXAMPLES = "256"
$env:NLP100_EPOCHS = "1"
python main.py 76
```

これらの制限を使用した精度は全データでの評価結果ではありません。最終結果を生成する
ときは環境変数を削除してから再実行してください。問題77はCUDA対応GPUが必須であり、
問題78および79はCUDAが利用可能な場合にGPUを自動的に使用します。

Google News Word2Vecのバイナリファイルは約3.4 GB、gzip圧縮ファイルは約1.6 GB
あります。短時間で動作確認する場合は、Word2Vecバイナリ形式の小規模なモデルと
評価件数の制限を利用してください。小規模モデルでは語彙や次元数が異なるため、
課題の最終結果にはGoogle Newsモデルを使用してください。

## AIの利用について

本課題ではCodexを利用しました。主な利用範囲は、`README.md`の作成、
ボイラープレートコードのリファクタリング、コードの検査、および
トラブルシューティングです。
