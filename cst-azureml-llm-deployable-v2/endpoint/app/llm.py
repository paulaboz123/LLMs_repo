import json
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

def make_client(endpoint: str, api_key: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_json(client: AzureOpenAI, deployment: str, system_prompt: str, user_prompt: str, temperature: float, top_p: float) -> dict:
    resp = client.chat.completions.create(
        model=deployment,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    return json.loads(content)
