import os
import streamlit as st
import uuid
import pprint
import datetime
import pytz
import io
from dotenv import load_dotenv
from datetime import timedelta
from string import Template
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, MomentoChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import LLMChain
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph


load_dotenv()

OUTPUT_PATH = "./output_data/report.pdf"

# チャット、レポート作成の機能をクラスで定義
# アンケートごとにインスタンスかを行う。
class Chat:
  def __init__(self, question):
    self.session_id = str(uuid.uuid4())
    self.question = question
    self.chat_log = []
    self.report = ""

  def talk(self, message, callback=StreamingStdOutCallbackHandler()):
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
        callbacks=[callback],
        )
    template = f"""
    あなたは市場調査をする調査員です以下のアンケートの回答に対する深掘りを行ってください。20字以内でアンケート：{self.question}
    """
    history = MomentoChatMessageHistory.from_client_params(
        self.session_id,
        os.environ["MOMENTO_CACHE"],
        timedelta(hours=int(os.environ["MOMENTO_TTL"])),
    )
    prompt = ChatPromptTemplate.from_messages([
      SystemMessagePromptTemplate.from_template(template),
      HumanMessagePromptTemplate.from_template("{input}"),
    ])
    memory = ConversationBufferMemory(chat_memory=history)
    chain = LLMChain(llm=chat, verbose=True, prompt=prompt, memory=memory)
    ai_message = chain.run(input=message)
    self._chat_log()
    self._output_log()
    return ai_message

  def refresh_session(self):
    self.session_id = str(uuid.uuid4())

  def make_report(self, title, output_path):
    text = "\n".join(message for message in self.chat_log)
    template="""
    次の文章は商品のアンケート結果です。
    {text}
    以下のフォーマットでレポートを作成してください。
    【アンケート実施日時】
    【アンケート内容】
    【回答と質問の一覧】
    【顧客の傾向】
    【顧客のインサイト】
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"]
    )
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = LLMChain(prompt=prompt, llm=chat)
    report = chain.run(text=text)
    self.report = report
    self._output_log()
    self._create_pdf(title, report, output_path)

  def _create_pdf(self, title, report, output_path):
    io_object = io.BytesIO()
    pdf = SimpleDocTemplate(io_object, pagesize=portrait(A4))
    styles = getSampleStyleSheet()
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
    styles['Normal'].fontName = 'HeiseiKakuGo-W5'
    styles['Normal'].fontSize = 12
    styles['Normal'].leading = 15  # 行間の設定
    title_style = ParagraphStyle(
      'Title',
      parent=styles['Normal'],
      fontSize=18,
      leading=50,
      alignment=1  # 中央揃え
    )
    title = Paragraph(title, title_style)
    para = Paragraph(report.replace("\n", "<br/>"), styles['Normal'])
    elements = [title, para]
    pdf.build(elements)
    pdf_data = io_object.getvalue()
    with open(output_path, 'wb') as file:
        file.write(pdf_data)

  def _get_history(self):
    history = MomentoChatMessageHistory.from_client_params(
      self.session_id,
      os.environ["MOMENTO_CACHE"],
      timedelta(hours=int(os.environ["MOMENTO_TTL"])),
    )
    conversationry = ConversationBufferMemory(chat_memory=history, return_messages=True)
    memory = conversationry.load_memory_variables({})["history"]
    return memory

  def _chat_log(self):
    chat_log = []
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    chat_log.append("日時" + now.strftime('%Y年%m月%d日 %H:%M:%S'))
    chat_log.append("アンケート：" + self.question)
    for index, chat_message in enumerate(self._get_history()):
      if type(chat_message) == HumanMessage:
        chat_log.append("回答：" + chat_message.content)
      elif type(chat_message) == AIMessage:
        chat_log.append("質問：" + chat_message.content)
    self.chat_log = chat_log

  def _output_log(self):
    print("【アンケート内容】\n" + self.question)
    print("【セッションID】\n" + self.session_id)
    print("【対話履歴】")
    print(self.chat_log)
    print("【レポート】" + self.report)


# サイドバー
st.sidebar.markdown("[使い方](https://github.com/dice-tfukuda/dice-demo/issues/1)")
question = st.sidebar.text_input('アンケートの内容を入力してください。', value="カレーの味はどうでしたか？")
title = st.sidebar.text_input('レポートのタイトルを入力してください。', value="アンケート調査まとめ")
if ("chat" not in st.session_state) or (st.session_state.question != question):
  st.session_state.question = question
  st.session_state.chat = Chat(question)

if st.sidebar.button('レポート作成'):
    st.session_state.chat.make_report(title, OUTPUT_PATH)
    if os.path.exists(OUTPUT_PATH):
      # ファイルをバイナリモードで読み込む
      with open(OUTPUT_PATH, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
      btn = st.sidebar.download_button(
          label="PDFをダウンロード",
          data=PDFbyte,
          file_name="report.pdf",
          mime='application/octet-stream'
      )
    else:
      st.write("ファイルが見つかりません。")

# 説明文
st.title("深掘りアプリデモ")
st.markdown(f"""
### アンケート：{question}
""")

# チャット本体
for index, chat_message in enumerate(st.session_state.chat._get_history()):
    if type(chat_message) == HumanMessage:
      with st.chat_message("user"):
        st.markdown(chat_message.content)
    elif type(chat_message) == AIMessage:
      with st.chat_message("assistant"):
        st.markdown(chat_message.content)

prompt = st.chat_input("What is up?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.chat.talk(prompt, callback)
        st.markdown(response)
