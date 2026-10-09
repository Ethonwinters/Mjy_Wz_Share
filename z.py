import os
import time
from openai import OpenAI

# 检查环境变量是否已设置
print("API Key 已设置：", bool(os.getenv("OPENAI_API_KEY")))
print("Base URL：", os.getenv("OPENAI_BASE_URL"))

# 创建客户端
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    timeout=60.0,
)

# 发送测试问题
start_time = time.perf_counter()

response = client.chat.completions.create(
    model="qwen38-27b-fp8",
    messages=[
        {
            "role": "system",
            "content": "你是一名动力总成技术专家。"
        },
        {
            "role": "user",
            "content": "请用100字解释什么是发动机爆震。"
        }
    ],
    temperature=0.1,
    max_tokens=512,
)

elapsed = time.perf_counter() - start_time

print("模型回答：")
print(response.choices[0].message.content)

print("\n模型名称：", response.model)
print("响应时间：", round(elapsed, 2), "秒")
print("Token 用量：", response.usage)
