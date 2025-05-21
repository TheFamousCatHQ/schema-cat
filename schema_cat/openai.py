import logging
import os
from xml.etree import ElementTree

from schema_cat.xml import xml_from_string
from schema_cat.prompt import build_system_prompt

logger = logging.getLogger("schema_cat")


async def call_openai(model: str,
                      sys_prompt: str,
                      user_prompt: str,
                      xml_schema: str,
                      max_tokens: int = 8192,
                      temperature: float = 0.0) -> ElementTree.XML:
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    system_prompt = build_system_prompt(sys_prompt, xml_schema)
    messages = [
        {"role": "system",
         "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()
    logger.info("Successfully received response from OpenAI")
    logger.debug(f"Raw response content: {content}")
    root = xml_from_string(content)
    logger.debug("Successfully parsed response as XML")
    return root
