# llm.py
import os
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:

        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # st.info(f"Using device: {device}") # 使用デバイスを表示
        device = 0 if torch.cuda.is_available() else -1 
        st.info(f"Using device: {'cuda' if device == 0 else 'cpu'}") # 使用デバイスを表示
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_auth_token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            # model=MODEL_NAME,
            model=model,
            model_kwargs={"torch_dtype": torch.float16},
            device=device,
            trust_remote_code=True
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        # messages = [
        #     {"role": "user", "content": user_question},
        # ]
        #stableLMはプレーンテキスト形式だそう
        prompt = f"ユーザー: {user_question}\nアシスタント:"
        # max_new_tokensを調整可能にする（例）⇒ modelの性能向上に合わせてトークン数を増やした
        # outputs = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
        outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9)

        # Gemmaの出力形式に合わせて調整が必要な場合がある - LM版に修正
        # 最後のassistantのメッセージを取得
        assistant_response = ""
        if outputs and isinstance(outputs, list) and isinstance(outputs[0].get("generated_text"), str):
            full_text = outputs[0]["generated_text"]
            
            if "アシスタント:" in full_text:
                assistant_response = full_text.split("アシスタント:")[-1].strip()
            else:
                assistant_response = full_text.strip()


        # if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
        #    if isinstance(outputs[0]["generated_text"], list) and len(outputs[0]["generated_text"]) > 0:
        #        # messages形式の場合
        #        last_message = outputs[0]["generated_text"][-1]
        #        if last_message.get("role") == "assistant":
        #            assistant_response = last_message.get("content", "").strip()
        #    elif isinstance(outputs[0]["generated_text"], str):
        #        # 単純な文字列の場合（古いtransformers？） - プロンプト部分を除く処理が必要かも
        #        # この部分はモデルやtransformersのバージョンによって調整が必要
        #        full_text = outputs[0]["generated_text"]
        #        # 簡単な方法：ユーザーの質問以降の部分を取得
        #        prompt_end = user_question
        #        response_start_index = full_text.find(prompt_end) + len(prompt_end)
        #        # 応答部分のみを抽出（より堅牢な方法が必要な場合あり）
        #        possible_response = full_text[response_start_index:].strip()
        #        # 特定の開始トークンを探すなど、モデルに合わせた調整
        #        if "<start_of_turn>model" in possible_response:
        #             assistant_response = possible_response.split("<start_of_turn>model\n")[-1].strip()
        #        else:
        #             assistant_response = possible_response # フォールバック

        if not assistant_response:
             # 上記で見つからない場合のフォールバックやデバッグ
             print("Warning: Could not extract assistant response. Full output:", outputs)
             assistant_response = "回答の抽出に失敗しました。"


        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s") # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        # エラーの詳細をログに出力
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0