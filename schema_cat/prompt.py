def build_system_prompt(sys_prompt: str, xml_schema: str = None) -> str:
    if xml_schema is None:
        return sys_prompt
    return sys_prompt + "\n\nReturn the results in XML format using the following structure. Only return the XML, nothing else.\n\n" + xml_schema
