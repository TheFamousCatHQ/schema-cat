# Enhanced Model Routing System

Schema Cat now includes an intelligent model routing system that provides flexible, configurable model selection with support for simple names, fuzzy matching, and advanced routing strategies.

## Key Features

### 🎯 Simple Model Names
Use friendly names instead of complex model identifiers:
```python
# Instead of complex names
result = await prompt_with_schema(prompt, UserInfo, "gpt-4.1-mini")

# Use simple names
result = await prompt_with_schema(prompt, UserInfo, "gpt4")
result = await prompt_with_schema(prompt, UserInfo, "claude")
result = await prompt_with_schema(prompt, UserInfo, "gemini")
```

### 🔍 Fuzzy Matching
The system can intelligently match partial or misspelled model names:
```python
# All of these resolve to appropriate models
result = await prompt_with_schema(prompt, UserInfo, "gpt4turbo")  # -> gpt-4.1-mini
result = await prompt_with_schema(prompt, UserInfo, "claudesonnet")  # -> claude-3.5-sonnet
result = await prompt_with_schema(prompt, UserInfo, "geminiflash")  # -> gemini-2.5-flash-preview
```

### ⚙️ Advanced Routing Strategies
Choose how models are selected based on your needs:
```python
from schema_cat import RoutingStrategy, ModelRequirements

# Use the cheapest available model
result = await prompt_with_schema(
    prompt, UserInfo, "gpt4",
    routing_strategy=RoutingStrategy.CHEAPEST
)

# Use the highest quality model
result = await prompt_with_schema(
    prompt, UserInfo, "claude",
    routing_strategy=RoutingStrategy.HIGHEST_QUALITY
)

# Specify model requirements
requirements = ModelRequirements(
    min_context_length=32000,
    supports_function_calling=True,
    max_cost_per_1k_tokens=0.01
)

result = await prompt_with_schema(
    prompt, UserInfo, "gpt4",
    model_requirements=requirements
)
```

### 🔧 Configuration-Based Overrides
Customize routing behavior through configuration files:

```yaml
# schema_cat_config.yaml
model_routing:
  default_strategy: "cheapest"
  
  aliases:
    my-gpt: "gpt-4.1-mini"
    my-claude: "claude-3.5-sonnet"
  
  provider_priority:
    - openrouter
    - openai
    - anthropic
  
  overrides:
    "gpt-4.1-mini":
      preferred_provider: "openai"
      strategy: "fastest"
```

## Usage Examples

### Basic Usage (Backward Compatible)
All existing code continues to work unchanged:
```python
from schema_cat import prompt_with_schema, Provider

# Existing code works exactly the same
result = await prompt_with_schema(
    prompt="Extract user info",
    schema=UserInfo,
    model="gpt-4.1-mini",
    provider=Provider.OPENAI
)
```

### Enhanced Usage with Simple Names
```python
from schema_cat import prompt_with_schema

# Use simple, memorable names
result = await prompt_with_schema(
    prompt="Extract user info",
    schema=UserInfo,
    model="gpt4"  # Automatically resolves to best GPT-4 variant
)

result = await prompt_with_schema(
    prompt="Analyze sentiment",
    schema=SentimentAnalysis,
    model="claude"  # Automatically resolves to best Claude variant
)
```

### Advanced Routing with Requirements
```python
from schema_cat import prompt_with_schema, ModelRequirements, RoutingStrategy

# For tasks requiring large context
requirements = ModelRequirements(min_context_length=100000)
result = await prompt_with_schema(
    prompt=long_document,
    schema=DocumentSummary,
    model="gpt4",
    model_requirements=requirements
)

# For cost-sensitive applications
result = await prompt_with_schema(
    prompt="Simple classification",
    schema=Classification,
    model="gpt4",
    routing_strategy=RoutingStrategy.CHEAPEST
)

# With provider preferences
from schema_cat import Provider
result = await prompt_with_schema(
    prompt="Generate content",
    schema=Content,
    model="claude",
    preferred_providers=[Provider.ANTHROPIC, Provider.OPENROUTER]
)
```

### Configuration Management
```python
from schema_cat import configure_routing, RouterConfig, load_config_from_file

# Load configuration from file
load_config_from_file("my_config.yaml")

# Or configure programmatically
config = RouterConfig(
    default_strategy=RoutingStrategy.CHEAPEST,
    aliases={"my-model": "gpt-4.1-mini"},
    provider_priority=[Provider.OPENROUTER, Provider.OPENAI]
)
configure_routing(config)
```

## Utility Functions

### Check Model Availability
```python
from schema_cat import get_available_models, validate_model_availability, get_provider_for_model

# Get all available models
models = get_available_models()
print(f"Available models: {models}")

# Check specific model
if validate_model_availability("gpt4"):
    print("GPT-4 is available")

# See which provider would be used
provider = get_provider_for_model("claude")
print(f"Claude would use: {provider}")
```

### Filter by Provider
```python
from schema_cat import get_available_models, Provider

# Get models available through OpenAI
openai_models = get_available_models(Provider.OPENAI)

# Get models available through OpenRouter
openrouter_models = get_available_models(Provider.OPENROUTER)
```

## Configuration File Format

### YAML Configuration
```yaml
model_routing:
  default_strategy: "best_available"
  
  aliases:
    gpt4: "gpt-4.1-mini"
    claude: "claude-3.5-sonnet"
    gemini: "gemini-2.5-flash-preview"
  
  provider_priority:
    - openrouter
    - openai
    - anthropic
  
  overrides:
    "gpt-4.1-mini":
      preferred_provider: "openai"
      fallback_providers: ["openrouter"]
      strategy: "fastest"
    
    "claude-3.5-sonnet":
      preferred_provider: "anthropic"
      cost_threshold: 0.01
```

### JSON Configuration
```json
{
  "model_routing": {
    "default_strategy": "best_available",
    "aliases": {
      "gpt4": "gpt-4.1-mini",
      "claude": "claude-3.5-sonnet"
    },
    "provider_priority": ["openrouter", "openai", "anthropic"],
    "overrides": {
      "gpt-4.1-mini": {
        "preferred_provider": "openai",
        "strategy": "fastest"
      }
    }
  }
}
```

## Configuration Loading

The system automatically looks for configuration files in this order:
1. `SCHEMA_CAT_CONFIG` environment variable
2. `schema_cat_config.json`
3. `.schema_cat.json`
4. `schema_cat_config.yaml` (if PyYAML is installed)
5. `.schema_cat.yaml` (if PyYAML is installed)

## Routing Strategies

| Strategy | Description |
|----------|-------------|
| `BEST_AVAILABLE` | Use the highest priority available model (default) |
| `CHEAPEST` | Select the most cost-effective option |
| `FASTEST` | Choose the model with lowest latency |
| `MOST_RELIABLE` | Pick the most stable/reliable provider |
| `HIGHEST_QUALITY` | Select the highest quality model available |

## Model Requirements

Specify requirements to filter available models:

```python
from schema_cat import ModelRequirements

requirements = ModelRequirements(
    min_context_length=32000,        # Minimum context window
    supports_function_calling=True,  # Must support function calling
    max_cost_per_1k_tokens=0.01,    # Maximum cost per 1k tokens
    min_quality_score=0.8            # Minimum quality score
)
```

## Migration Guide

### From Static MODEL_PROVIDER_MAP
The old static mapping is still supported but you can now use the enhanced system:

**Before:**
```python
# Had to know exact model names and manage fallbacks manually
result = await prompt_with_schema(prompt, UserInfo, "gpt-4.1-mini", Provider.OPENAI)
```

**After:**
```python
# Simple, intuitive names with automatic provider selection
result = await prompt_with_schema(prompt, UserInfo, "gpt4")
```

### Gradual Migration
You can migrate gradually by setting `use_smart_routing=False` to use the legacy system:

```python
# Use legacy routing for specific calls
result = await prompt_with_schema(
    prompt, UserInfo, "gpt-4.1-mini",
    use_smart_routing=False
)
```

## Benefits

1. **Simplified API**: Use memorable names like "gpt4" instead of complex identifiers
2. **Intelligent Fallbacks**: Automatic provider selection based on availability
3. **Cost Optimization**: Choose routing strategies based on your priorities
4. **Flexible Configuration**: Customize behavior without code changes
5. **Fuzzy Matching**: Handles typos and partial names gracefully
6. **Full Backward Compatibility**: Existing code continues to work unchanged
7. **Provider Agnostic**: Easy to switch between providers or add new ones

## Troubleshooting

### No Models Available
If you get "No available provider" errors:
1. Check that you have API keys set for at least one provider
2. Verify the model name is correct or try a simpler alias
3. Use `get_available_models()` to see what's available

### Configuration Not Loading
1. Check file path and format
2. Ensure PyYAML is installed for YAML files: `pip install pyyaml`
3. Validate JSON syntax for JSON files
4. Check the `SCHEMA_CAT_CONFIG` environment variable

### Fuzzy Matching Issues
If fuzzy matching isn't working as expected:
1. Try more specific model names
2. Check available aliases with the registry
3. Use exact canonical names for precise control

## Advanced Features

### Custom Model Registration
```python
from schema_cat.model_registry import get_global_registry, ModelInfo, ModelCapabilities

registry = get_global_registry()

# Register a custom model
registry.register_model(
    canonical_name="my-custom-model",
    provider=Provider.OPENAI,
    provider_model_name="gpt-4-custom",
    aliases=["custom", "my-gpt"],
    capabilities=ModelCapabilities(
        context_length=128000,
        supports_function_calling=True,
        cost_per_1k_tokens=0.03
    )
)
```

### Dynamic Provider Selection
```python
from schema_cat.model_router import get_global_router

router = get_global_router()

# Get detailed routing information
route_result = router.route_model("gpt4")
if route_result:
    print(f"Provider: {route_result.resolution.provider}")
    print(f"Model: {route_result.resolution.model_name}")
    print(f"Reason: {route_result.routing_reason}")
    print(f"Confidence: {route_result.resolution.confidence}")
```

This enhanced routing system provides a much more flexible and user-friendly way to work with multiple AI providers while maintaining full backward compatibility with existing code.