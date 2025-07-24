import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def handler(event, context):
    """Netlify Functionのメインハンドラ"""
    # APIキーを環境変数から読み込む (NetlifyのUIで設定)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"statusCode": 500, "body": json.dumps({"error": "APIキーが設定されていません。"})}

    # POSTリクエスト以外は無視
    if event['httpMethod'] != 'POST':
        return {"statusCode": 405, "body": "Method Not Allowed"}

    try:
        body = json.loads(event['body'])
        question = body.get('question')
        if not question:
            return {"statusCode": 400, "body": json.dumps({"error": "質問がありません。"})}

        # --- RAGのセットアップ ---
        # __file__は現在のファイルパス、os.path.dirnameでディレクトリを取得
        # これにより、データベースの場所を正しく見つける
        current_dir = os.path.dirname(__file__)
        db_directory = os.path.join(current_dir, '..', '..', 'chroma_db')

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=api_key)

        prompt_template_str = """
        提供された「関連情報」だけを使って、ユーザーの「質問」に日本語で詳しく回答してください。
        # 関連情報:
        {context}
        # 質問:
        {question}
        # 回答:
        """
        prompt = PromptTemplate.from_template(prompt_template_str)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # RAGチェーンを実行して回答を生成
        answer = rag_chain.invoke(question)
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"エラーが発生しました: {str(e)}"})
        }