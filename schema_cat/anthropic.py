import logging
import os
from xml.etree import ElementTree

from schema_cat.xml import xml_from_string

logger = logging.getLogger("schema_cat")


async def call_anthropic(model: str,
                         sys_prompt: str,
                         user_prompt: str,
                         xml_schema: str,
                         max_tokens: int = 8192,
                         temperature: float = 0.0) -> ElementTree.XML:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    system_prompt = sys_prompt + "\n\nReturn the results in XML format using the following structure:\n\n" + xml_schema
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    content = response.content[0].text.strip() if hasattr(response.content[0], 'text') else response.content[0][
        'text'].strip()
    logger.info("Successfully received response from Anthropic")
    logger.debug(f"Raw response content: {content}")
    root = xml_from_string(content)
    logger.debug("Successfully parsed response as XML")
    return root
