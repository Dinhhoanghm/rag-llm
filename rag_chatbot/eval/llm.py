import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenRouterLLM:
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def query(self, system_prompt: str, messages: list[dict] = []) -> str:
        messages = messages or []
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}] + [],
            temperature=self.temperature,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    llm = OpenRouterLLM(model_name="qwen/qwen-2.5-72b-instruct:free")
    print(llm.query("Hello, how are you?", []))
