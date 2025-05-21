import logging
import os
from xml.etree import ElementTree

import httpx

from schema_cat.xml import xml_from_string
from schema_cat.prompt import build_system_prompt

logger = logging.getLogger("schema_cat")


async def call_openrouter(model: str,
                          sys_prompt: str,
                          user_prompt: str,
                          xml_schema: str,
                          max_tokens: int = 8192,
                          temperature: float = 0.0) -> ElementTree.XML:
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    system_prompt = build_system_prompt(sys_prompt, xml_schema)
    data = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://www.thefamouscat.com"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "SchemaCat"),
        "Content-Type": "application/json"
    }

    logger.info(f"Calling OpenRouter API directly with model {model}")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

    logger.info("Successfully received response from OpenRouter")
    logger.debug(f"Raw response content: {content}")

    # Parse the response content as XML
    root = xml_from_string(content)
    logger.debug("Successfully parsed response as XML")
    return root
