import getpass
import os

from openai import OpenAI


def _set_env(key: str) -> None:
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")


_set_env("OPENAI_API_KEY")

os.environ["OPENAI_BASE_URL"] = (
    "https://eisapi.byd.com/open-api/1.0/llm/v1/qwen38-27b-fp8"
)

client = OpenAI()

response = client.chat.completions.create(
    model="qwen38-27b-fp8",
    messages=[
        {
            "role": "system",
            "content": "你是一个专业、严谨且友好的中文助手。请使用简洁、通俗易懂的语言回答问题。",
        },
        {
            "role": "user",
            "content": "请用通俗易懂的语言解释为什么天空是蓝色的。",
        },
    ],
)

print(response.choices[0].message.content)
