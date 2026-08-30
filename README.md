# 言語処理100本ノック 2025 — 第6章

富士フイルム自然言語処理課題コードです。

## 実行方法

### 環境設定

Pythonの仮想環境を作成し、依存パッケージをインストールします。以下は
PowerShellで実行する例です。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

以降のコマンドは、プロジェクトのルートディレクトリで実行してください。

### コマンドから実行

`main.py`に第6章の問題番号（50〜59）を渡して実行します。

```powershell
python main.py 50
python main.py 54
python main.py 55
```

`main.py`は対象問題の`PROBLEM_DESCRIPTION`を表示してから、登録された
`main()`を現在のPythonプロセス内で呼び出します。サブプロセスは使用しません。

### VS Codeから実行

1. プロジェクトのルートディレクトリをVS Codeで開きます。
2. コマンドパレットの「Python: Select Interpreter」から
   `venv\Scripts\python.exe`を選択します。
3. 「実行とデバッグ」ビューで`Pydbg: main.py`を選択して開始します。
4. 引数入力欄に問題番号を入力します。例: `54`

デバッグ設定は[`.vscode/launch.json`](.vscode/launch.json)にあります。環境変数を
利用する場合は、プロジェクトルートに`.env`を作成すると、この設定から読み込まれます。

## コード構成

```text
.
├── main.py              # 問題番号から実行する関数を選択するエントリーポイント
├── chap06/              # 第6章: 単語ベクトル（問題50〜59）
│   ├── 50.py〜59.py     # 各問題の実装
│   └── common.py        # 第6章で共通して利用する処理
├── utils/               # パス、キャッシュ、出力などの共通処理
├── res/                 # データセットおよび単語ベクトルのキャッシュ
└── out/                 # 問題ごとの実行結果
```

各問題は`chap06/50.py`から`chap06/59.py`に実装され、先頭に
`PROBLEM_DESCRIPTION`、実行処理として`main()`を持ちます。第6章内の共通処理は
`chap06/common.py`、汎用的な処理は`utils/`に配置しています。

問題55は問題54の実行結果を利用します。`out/54/capital_common_countries.tsv`が
存在する場合は問題54を再実行せず、存在しない場合に限り問題54の`main()`を
同じプロセス内で実行します。問題54の対象は`capital-common-countries`セクション
のみであるため、この結果に文法的アナロジーの事例は含まれません。

## `res`と`out`

`res/<データセット名>/`には、ダウンロードした評価データとモデルキャッシュを
保存します。主なデータは次のとおりです。

- `res/questions-words/questions-words.txt`: 単語アナロジー評価データ
- `res/WordSimilarity-353/wordsim353.tsv`: WordSimilarity-353評価データ
- `res/gensim-data/`: gensimが取得した単語ベクトルのキャッシュ

`out/<問題番号>/`には、各問題が生成したJSON、TSV、画像などを保存します。
例えば、問題54の結果は`out/54/`、問題58と59の可視化結果はそれぞれ
`out/58/`と`out/59/`に保存されます。

`res/`と`out/`の内容はGitの管理対象外です。必要に応じて再生成してください。

## 環境変数

環境変数はPowerShellで設定するか、VS Codeから実行する場合は`.env`に記述します。

| 環境変数 | デフォルト値 | 説明 |
| --- | --- | --- |
| `NLP100_WORD_VECTORS` | `word2vec-google-news-300` | 第6章で使用する単語ベクトル |
| `NLP100_ANALOGY_LIMIT` | 制限なし | analogyデータで各セクションから使用する最大事例数 |
| `NLP100_RESTRICT_VOCAB` | 制限なし | 類似語検索で使用する語彙数 |

例:

```powershell
$env:NLP100_WORD_VECTORS = "glove-wiki-gigaword-50"
$env:NLP100_ANALOGY_LIMIT = "10"
$env:NLP100_RESTRICT_VOCAB = "50000"
python main.py 54
```

デフォルトのGoogle News Word2Vecモデルは約1.6 GBあります。短時間で動作確認する
場合は、軽量な単語ベクトルと評価件数の制限を利用してください。軽量モデルでは
Google Newsモデルと語彙や次元数が異なるため、課題の最終結果にはデフォルトモデルを
使用してください。
