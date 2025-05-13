# prompt.py

from typing import List
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_dummy_topic(
    name: str, department: str, hobby: List[str], hometown: str
) -> str:
    return (
        f"{name}さんにおすすめの話題：最近、{department}で{', '.join(hobby)}が話題です！"
        f" また、{hometown}出身の人たちとの交流も盛んですよ。"
    )


def generate_openai_topic(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    message = response.choices[0].message
    content = message.content

    if content is None:
        raise ValueError("OpenAI returned no content.")

    return content
