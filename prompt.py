# prompt.py

from typing import List
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY") # .envのAPIキーを取得して利用

def generate_dummy_topic(
    name: str, department: str, hobby: List[str], hometown: str
) -> str:
    return (
        f"{name}さんにおすすめの話題：最近、{department}で{', '.join(hobby)}が話題です！"
        f" また、{hometown}出身の人たちとの交流も盛んですよ。"
    )


def generate_openai_topic(prompt: str) -> str:
     response = openai.ChatCompletion.create( # type: ignore
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
     )
     return response.choices[0].message.content
