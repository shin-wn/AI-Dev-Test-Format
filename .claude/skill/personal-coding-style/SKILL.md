---
name: personal-coding-style
description: Python コードを書く・編集する際に必ず適用するコーディング規約。命名（平易な英単語）、コメントの位置と具体性、空行、後方互換性の禁止、定数の作り方、ハードコード回避、関数分割の判断基準、エントリポイントの作り方（__main__.py 禁止）。Python コードの新規作成・修正の前に必ず読むこと。
---

# コーディング規約 — vibe coding の典型的な失敗を防ぐ

コードを書く前にこの規約を適用する。すべてのルールは「後から人間（日本人エンジニア）が精読・改修する」ことを前提にしている。

## 1. 命名 — 平易な英単語・略さない・意味を一致させる

- **中学〜高校レベルの平易な英単語を選ぶ。** 非ネイティブが辞書なしで読めること。ネイティブには自然でも日本人に馴染みの薄い語は避ける。
  - ❌ `coalesce`, `collate`, `munge`, `elide`, `stem`, `probe`, `hydrate`, `shim`
  - ✅ `merge`, `sort_and_group`, `clean`, `remove`, `shorten`, `check`, `fill`, `wrapper`
- **略語を作らない。** 広く通じるもの（`id`, `url`, `max`, `min`, `num`, `dir`, `config`, `img`, `db`）のみ許可。それ以外はフルスペル。
  - ❌ `res`, `ctx`, `mgr`, `proc`, `cnt`, `tmpl`, `gen_cfg`
  - ✅ `result`, `context`, `manager`, `count`, `template`, `generation_config`
- **抽象的すぎる名前を禁止。** 中身が特定できる具体名にする。
  - ❌ `data`, `info`, `item`, `obj`, `handler`, `process()`, `do_task()`
  - ✅ `user_rows`, `error_messages`, `image_paths`, `remove_expired_sessions()`
- **名前と実際の処理・内容を一致させる。** 処理を変更したら名前も追従させる。
  - ❌ フィルタ処理も含むのに `load_users()`
  - ✅ `load_and_filter_users()`、あるいは読み込みとフィルタを分ける

## 2. コメント — 場所と具体性

**書かない場所:**
- モジュール最上部の説明コメント・モジュール docstring（原則不要。書くなら1行まで）
- コードを読めば自明な行の逐語訳コメント（`# インクリメントする` など）

**必ず書く場所:**
- 関数・クラス・メソッドの docstring — 何をするか、引数と戻り値の意味
- 処理ブロックの区切り — そのブロックが何をしているか
- スキーマ・dataclass・TypedDict・Pydantic モデルの**各フィールド**の説明

**具体性:** 抽象的な要約ではなく、判断基準・値の意味まで書く。

```python
# ❌ 抽象的
def select_results(results):
    """結果を処理して返す"""

# ✅ 具体的
def select_results(results: list[ScoredResult], threshold: float) -> list[str]:
    """スコア降順に並べ、threshold 未満の結果を除外して名前のみ返す。

    Args:
        results: 検索が返した候補とスコアのペア
        threshold: これ未満のスコアはノイズとみなして捨てる
    """
```

```python
class RetryConfig(BaseModel):
    max_attempts: int   # 初回実行を含む総試行回数（リトライ回数ではない）
    wait_seconds: float # 試行間の待ち時間。指数バックオフの初期値
    timeout: int        # 1試行あたりの上限秒数
```

## 3. 空行 — 処理の塊ごとに区切る

論理的なステップの区切りに空行を1行入れる。連続した処理をギチギチに詰めない。「1つの空行ブロック = 1つのやること」を保ち、ブロック先頭に必要なら説明コメントを置く。

```python
# ❌ 詰め込み
def export_reports(source_dir, output_dir):
    csv_paths = list(source_dir.glob("*.csv"))
    csv_paths = [p for p in csv_paths if p.stat().st_size > 0]
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_paths:
        meta_path = csv_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        shutil.copy(csv_path, output_dir)
        shutil.copy(meta_path, output_dir)

# ✅ ステップごとに区切る
def export_reports(source_dir: Path, output_dir: Path) -> None:
    # 空ファイルを除いた CSV を集める
    csv_paths = list(source_dir.glob("*.csv"))
    csv_paths = [p for p in csv_paths if p.stat().st_size > 0]

    output_dir.mkdir(parents=True, exist_ok=True)

    # メタデータが揃っている CSV だけをコピーする
    for csv_path in csv_paths:
        meta_path = csv_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        shutil.copy(csv_path, output_dir)
        shutil.copy(meta_path, output_dir)
```

## 4. 後方互換性を持たせない

指示がない限り、後方互換のためのコードを一切書かない。

- 旧シグネチャを残したラッパー関数・deprecated エイリアスを作らない
- 「古い形式も受け付ける」フォールバック分岐を作らない
- シグネチャや呼び出し方を変えたら、**呼び出し側をすべて書き換える**

外部利用者のいない個人〜単一チームの開発を前提とする。

## 5. `from __future__ import annotations` を書かない

Python（3.12 以降）を使う前提のため不要。新規モジュールに `from __future__ import annotations` を入れない。既存コードにあったら削除する。

## 6. モジュールレベル定数（ALL_CAPS）を安易に作らない

ALL_CAPS のグローバル定数を作ってよいのは:
- **複数の関数**から参照される真の定数（マジックナンバー・固定文字列の命名）のみ

それ以外は、エントリポイント（`main()` や最上位の関数）でローカル変数として作り、**引数で下に渡す**。

```python
# ❌ 1箇所でしか使わない値をグローバルに逃がす
DEFAULT_THRESHOLD = 0.35
OUTPUT_SUBDIR = "filtered"

def filter_results(results):
    return [r for r in results if r.score >= DEFAULT_THRESHOLD]

# ✅ 呼び出しフローで渡す
def main() -> None:
    threshold = 0.35
    results = filter_results(load_results(), threshold)

def filter_results(results: list[ScoredResult], threshold: float) -> list[ScoredResult]:
    return [r for r in results if r.score >= threshold]
```

グローバル定数は「設定値の隠れた注入経路」になり、関数の入出力を追いづらくする。迷ったら引数で渡す。

## 7. ハードコードで無理やり解決しない

設計が曖昧なとき・仕様の隙間に気づいたときに、特定ケースだけ通る決め打ち実装で埋めない。

- ❌ 特定のファイル名・キー名・インデックスへの決め打ち分岐（`if name == "sample01":` のような特殊ケース埋め込み）
- ❌ 「とりあえず今のデータでは動く」前提の暗黙の仮定
- ✅ 引数化・データ駆動（設定ファイル・辞書）にして特殊ケースをデータ側に追い出し、作業は止めない
- ✅ 置いた仮定を隠さない: 何をどう仮定したかを完了報告に明記する
- ✅ どの解釈を選ぶかで作りが大きく変わり、外れたら手戻りが大きい場合のみ、実装前にユーザーに確認する

「動くけど汎用性がない」実装で黙って埋めるくらいなら、汎用化して仮定を報告する方が正しい。

## 8. 関数・メソッド分割の判断基準

分割は目的ではない。**上から下に一直線に読めること**を最優先する。

**分割してよい:**
- 複数箇所から再利用される
- 独立してテストしたい単位である
- 分割することで呼び出し側が「何をしているかの目次」として読めるようになる

**分割しない:**
- 一度しか呼ばれず、分割すると読み手が定義へのジャンプを強いられるだけの場合 → 空行＋ブロックコメントで区切る（ルール3）
- 分割後の関数に引数を5個以上渡す必要がある場合（設計が悪いサイン。分割をやめるか設計を見直す）
- 数行の逐次処理を「短い関数が良い」という理由だけで切り出す場合

迷ったら分割しない。あとで再利用が発生してから切り出す方が安全。

## 9. エントリポイントは `cli.py` + `[project.scripts]` — `__main__.py` を作らない

パッケージの実行入口として `__main__.py` を作らない。名前付きモジュール `cli.py` に `main()` を置き、`pyproject.toml` の `[project.scripts]` にコマンド名で登録して `uv run <コマンド名>` で起動する。

```toml
[project.scripts]
コマンド名 = "パッケージパス.cli:main"
```

`__main__.py` を避ける理由:

- `python -m パッケージ` は「知っている人だけが打てる」暗黙の入口になる。`[project.scripts]` なら公開コマンドの一覧が pyproject.toml に目次として並ぶ
- dunder ファイル名は中身を表さず、「入口ファイル」という曖昧な立場のせいで実処理のロジックが溜まりやすい
- `cli.py` は普通のモジュールとして import してテストできる

`cli.py` の中身は薄く保つ: 引数パース＋各モジュールの関数を呼ぶ配線だけ。実処理は名前付きモジュールに置く。サブコマンドが必要なら argparse の subparsers を `cli.py` 内で使う。既存コードに `__main__.py` が残っていても真似しない。
