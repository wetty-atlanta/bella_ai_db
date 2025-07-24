import os
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 初期設定 ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("APIキーが設定されていません。Renderの環境変数を確認してください。")

# --- ベクトルデータベースの読み込み ---
db_directory = 'chroma_db'
if not os.path.exists(db_directory):
    raise RuntimeError(f"データベース '{db_directory}' が見つかりません。'setup_db.py' を実行してください。")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5}) # 関連性の高いチャンクを5つ検索

# --- AIモデルとプロンプトの設定 ---
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=api_key, temperature=0.7)

# プロンプトテンプレート
prompt_template = """
あなたは漫画『Bella』シリーズに関する、知識豊富な解説者です。
提供された「関連情報」だけを使って、ユーザーの「質問」に日本語で詳しく回答してください。
もし関連情報に答えがない場合は、「物語の中では言及されていないようです。」と回答してください。

# 関連情報:
{context}

# 質問:
{question}

# 回答:
"""
prompt = PromptTemplate.from_template(prompt_template)

# --- LangChainの処理チェーンを定義 ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Flaskアプリケーション ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_rag():
    data = request.get_json()
    question = data.get('question')
    # modeはRAGでは不要になるが、互換性のために残す
    mode = data.get('mode')

    if not question:
        return jsonify({"error": "質問を入力してください"}), 400

    try:
        # RAGチェーンを実行して回答を生成
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"回答の生成中にエラーが発生しました: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))