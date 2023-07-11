#!/usr/bin/env python

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def remove_files(directory, extension):
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(extension):
        file_path = os.path.join(root, file)
        os.remove(file_path)

def transcribe(youtube_video_url):
  loader = GenericLoader(YoutubeAudioLoader([youtube_video_url], "./tmp"), OpenAIWhisperParser())
  docs = loader.load()

  remove_files('./tmp', '.m4a')

  return ' '.join(map(lambda doc: doc.page_content, docs))

def summarise(transcription):
  chat = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0,max_tokens=1024)

  system_prompt = SystemMessagePromptTemplate.from_template("""
  下記SCRIPTは、アジャイル開発に関するカンファレンスの講演内容を書き起こしたものです。
  このあと、この講演内容に関して質問されます。

  == SCRIPT ==
  {transcription}
  """)
  human_prompt = HumanMessagePromptTemplate.from_template("""
  この講演の内容の要約を500字で作成してください。

  作成は下記のステップで行ってください。
  1. この講演で語られている、講演者の立場を特定。これをAとする。
  2. この講演で語られている、講演者が設定した課題を特定。これをBとする。
  3. この講演で語られている、課題Bに対してAが行った対応を特定。これをCとする。
  4. この講演で語られている、課題Bに対して対応Cを発見するに至ったきっかけを特定。これをDとする。
  5. この講演で語られている、対応Cによって課題Bが解決したことによりAが得た成果を特定。これをEとする。

  出力は、下記のフォーマットFORMATに従ってください。

  === FORMAT ===
  【要約】
  <<
  　下記の内容を含めて、この講演を紹介する説明文を作成。
  　・課題Bのうち主要な課題を解決しようとした経緯
  　・その主要な課題を解決するためにAが行った対応C
  　・その主要な課題が解決されたことにより得られた成果E
  >>

  【ハッシュタグ】
  <<
  　下記の指示に従ってハッシュタグを生成。
  　・講演者Aの属性を表すものを1つ
  　・課題Bのうち主要な課題を表すものを1つ
  　・対応Cのうち主要な課題を解決するものを1つ
  >>

【各情報】
  <<AからEまでの情報を出力>>
  """)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
  chain = LLMChain(llm=chat, prompt=chat_prompt,verbose=True)

  return chain.run(transcription=transcription)

def main():
  youtube_video_url = sys.argv[1]

  transcription = transcribe(youtube_video_url=youtube_video_url)
  summary = summarise(transcription)

  print(summary)


if __name__ == "__main__":
  main()
