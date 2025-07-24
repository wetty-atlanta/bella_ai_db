import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- 初期設定 ---
# 環境変数からAPIキーを読み込みます
# このスクリプトを実行する前に、ターミナルでAPIキーを設定してください
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("APIキーが設定されていません。環境変数 'GOOGLE_API_KEY' を設定してください。")

# --- ファイルの読み込み ---
def load_file(filepath):
    """指定されたパスのテキストファイルを読み込む"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"警告: {filepath} が見つかりませんでした。スキップします。")
        return None

print("プロットファイルを読み込んでいます...")
# ファイル名を修正: Bella.txt -> bella_main.txt
main_plot = load_file('bella_main.txt')
sequel_plot = load_file('plot.txt')
prequel_plot = load_file('Original.txt')

# 読み込んだテキストを結合
all_texts = []
if main_plot: all_texts.append(main_plot)
if sequel_plot: all_texts.append(sequel_plot)
if prequel_plot: all_texts.append(prequel_plot)

if not all_texts:
    raise RuntimeError("読み込むプロットファイルが一つも見つかりませんでした。処理を中断します。")

# --- テキストの分割（チャンキング） ---
print("テキストを小さなチャンクに分割しています...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.create_documents(all_texts)
print(f"合計 {len(documents)} 個のチャンクに分割されました。")

# --- ベクトルデータベースの作成 ---
print("ベクトルデータベースを作成しています...（数分かかる場合があります）")
# GeminiのEmbeddingモデルを使用
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# データベースを保存するディレクトリ
db_directory = 'chroma_db'

# ChromaDBにドキュメントを保存
db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=db_directory
)

print(f"データベースの作成が完了しました。'{db_directory}' ディレクトリに保存されました。")
print("このディレクトリをGitHubリポジトリに追加してプッシュしてください。")