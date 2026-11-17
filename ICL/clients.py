import openai

BASE_URL = "https://llmapi.blsc.cn"
# MODEL = "DeepSeek-V4.1-Flash-1017"
# MODEL = "GPT-5.6-Sol"
MODEL = "Qwen3.8-Max"
api_key = "sk-6HkvAUux0Q6acFc6FVB41w"


def chat_llm(prompt) -> str:
    client = openai.OpenAI(
        api_key=api_key,
        base_url=BASE_URL + "/v1/",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Follow the requested machine-readable schema exactly. "
                "Do not add prose outside the requested output tags."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(model=MODEL,
                                                  # reasoning_effort="medium",
                                                  reasoning_effort="none",
                                                  response_format={"type": "json_object"},
                                                  stream=False,
                                                  messages=messages)
        content = response.choices[0].message.content
        return content.strip() or "{}"
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}") from e
