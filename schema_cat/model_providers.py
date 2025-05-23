from schema_cat.provider_enum import Provider

# Canonical model mapping: maps internal model names to provider/model pairs in order of preference
MODEL_PROVIDER_MAP = {
    # Google Gemini
    "gemini-2.5-flash-preview": [
        (
            Provider.OPENROUTER,
            "google/gemini-2.5-flash-preview",
        ),  # Default, categorize_files_openrouter, etc.
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.5-sonnet-latest"),
    ],
    # OpenAI nano
    "gpt-4.1-nano-2025-04-14": [
        (
            Provider.OPENROUTER,
            "openai/gpt-4.1-nano-2025-04-14",
        ),  # categorize_files_openai_json
        (Provider.OPENAI, "gpt-4.1-nano-2025-04-14"),
        (Provider.ANTHROPIC, "claude-3.5-sonnet-latest"),
    ],
    # OpenAI mini
    "gpt-4.1-mini": [
        (Provider.OPENROUTER, "openai/gpt-4.1-mini"),  # bug_analyzer (schema_cat)
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    "openai/gpt-4.1-mini": [
        (Provider.OPENROUTER, "openai/gpt-4.1-mini"),
        (Provider.OPENAI, "gpt-4.1-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    # OpenAI gpt-4o-mini
    "gpt-4o-mini": [
        (Provider.OPENROUTER, "openrouter/gpt-4o-mini"),  # validate_complexity_report
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    "openrouter/gpt-4o-mini": [
        (Provider.OPENROUTER, "openrouter/gpt-4o-mini"),
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3.7-sonnet-latest"),
    ],
    # Anthropic Claude Sonnet
    "claude-3.5-sonnet": [
        (
            Provider.ANTHROPIC,
            "claude-3.5-sonnet-latest",
        ),  # Docstring reference in create_agent
        (Provider.OPENROUTER, "anthropic/claude-3.5-sonnet"),
        (Provider.OPENAI, "anthropic/gpt-4.1-mini"),
    ],
    "anthropic/claude-3.5-sonnet": [
        (Provider.ANTHROPIC, "claude-3.5-sonnet-latest"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-sonnet"),
        (Provider.OPENAI, "anthropic/gpt-4.1-mini"),
    ],
    # Existing entries
    "claude-haiku": [
        (Provider.ANTHROPIC, "claude-3-haiku-20240307"),
        (Provider.OPENROUTER, "openrouter/claude-3-haiku-20240307"),
        (Provider.OPENAI, "gpt-4.1-nano"),  # fallback to a similar OpenAI model
    ],
    "anthropic/claude-3.5-haiku": [
        (Provider.ANTHROPIC, "claude-3-5-haiku-latest"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-haiku"),
        (Provider.OPENAI, "anthropic/gpt-4.1-nano"),
    ],
    "gemma": [
        (Provider.OPENROUTER, "google/gemma-3-4b-it"),
        (Provider.OPENROUTER, "anthropic/claude-3.5-haiku"),
        (Provider.ANTHROPIC, "claude-3-5-haiku-latest"),
    ],
    "claude-sonnet-4": [
        (Provider.ANTHROPIC, "claude-sonnet-4-20250514"),
        (Provider.OPENROUTER, "anthropic/claude-sonnet-4")
    ]
}
