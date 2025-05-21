def build_system_prompt(sys_prompt: str, xml_schema: str) -> str:
    return sys_prompt + "\n\nReturn the results in XML format using the following structure. Only return the XML, nothing else.\n\n" + xml_schema
