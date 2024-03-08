import streamlit as st

st.title("DiCEのデモ環境")

# READMEファイルのパスを設定
readme_path = './README.md'

# ファイルを開き、内容を読み込む
with open(readme_path, 'r', encoding='utf-8') as file:
    readme_content = file.read()

# READMEの内容を表示（オプション）
st.markdown(f"""
{readme_content}
""")