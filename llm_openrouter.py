import click
from enum import Enum
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat, Completion
from pathlib import Path
from pydantic import Field, field_validator
from typing import Optional, Union
import json
import time
import httpx
from typing import Optional
from pydantic import Field

def get_openrouter_models():
    models = fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=3600,
    )["data"]
    return models

def load_prefill(name: str) -> Optional[str]:
    """Load a prefill from the prefills directory"""
    path = llm.user_dir() / "prefills" / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    return None

def get_supports_images(model_definition):
    try:
        return "image" in model_definition["architecture"]["input_modalities"]
    except KeyError:
        return False


def has_parameter(model_definition, parameter):
    try:
        return parameter in model_definition["supported_parameters"]
    except KeyError:
        return False


class ReasoningEffortEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class _mixin:
    class Options(Chat.Options):
        online: Optional[bool] = Field(
            description="Use relevant search results from Exa",
            default=None,
        )
        provider: Optional[Union[dict, str]] = Field(
            description=("JSON object to control provider routing"),
            default=None,
        )
        reasoning_effort: Optional[ReasoningEffortEnum] = Field(
            description='One of "high", "medium", or "low" to control reasoning effort',
            default=None,
        )
        reasoning_max_tokens: Optional[int] = Field(
            description="Specific token limit to control reasoning effort",
            default=None,
        )
        reasoning_enabled: Optional[bool] = Field(
            description="Set to true to enable reasoning with default parameters",
            default=None,
        )

        @field_validator("provider")
        def validate_provider(cls, provider):
            if provider is None:
                return None

            if isinstance(provider, str):
                try:
                    return json.loads(provider)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in provider string")
            return provider

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("provider", None)
        kwargs.pop("online", None)
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning_max_tokens", None)
        kwargs.pop("reasoning_enabled", None)
        extra_body = {}
        if prompt.options.online:
            extra_body["plugins"] = [{"id": "web"}]
        if prompt.options.provider:
            extra_body["provider"] = prompt.options.provider
        reasoning = {}
        if prompt.options.reasoning_effort:
            reasoning["effort"] = prompt.options.reasoning_effort
        if prompt.options.reasoning_max_tokens:
            reasoning["max_tokens"] = prompt.options.reasoning_max_tokens
        if prompt.options.reasoning_enabled is not None:
            reasoning["enabled"] = prompt.options.reasoning_enabled
        if reasoning:
            extra_body["reasoning"] = reasoning
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs


class OpenRouterChat(_mixin, Chat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


class OpenRouterAsyncChat(_mixin, AsyncChat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    class Options(Chat.Options):
        prefill: Optional[str] = Field(
            description="Initial text for the model's response. Uses OpenRouter's Assistant Prefill feature.",
            default=None
        )
        pre: Optional[str] = Field(
            description="Load prefill from a saved file in the prefills directory",
            default=None
        )
        online: bool = Field(
            description="Enable web browsing capability via OpenRouter",
            default=False
        )

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)

    def execute(self, prompt, stream, response, conversation, key=None):
        messages = self._build_messages(conversation, prompt)
        response._prompt_json = {"messages": messages}
        kwargs = self.build_kwargs(prompt)

        # Remove prefill and pre from kwargs since we handle them in messages
        kwargs.pop('prefill', None)
        kwargs.pop('pre', None)
        kwargs.pop('online', None)  # Remove online from kwargs

        client = self.get_client(key)

        # Append :online to model name if online capability is enabled
        model_name = self.model_name
        if prompt.options.online:
            model_name = f"{model_name}:online"

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=stream,
                **kwargs,
            )

            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

            response.response_json = {"content": "".join(response._chunks)}
        except Exception as e:
            raise llm.ModelError(f"OpenRouter API error: {str(e)}")

    def _build_messages(self, conversation, prompt):
        """Build the messages list for the API call."""
        messages = []
        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        # Add system message if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        messages.append({"role": "user", "content": prompt.prompt})

        # Add prefill if specified
        if prompt.options.pre:
            prefill = load_prefill(prompt.options.pre)
            if prefill:
                messages.append({
                    "role": "assistant",
                    "content": prefill
                })
        elif prompt.options.prefill:
            messages.append({
                "role": "assistant",
                "content": prompt.options.prefill
            })

        return messages

    def build_kwargs(self, prompt):
        # Return any additional kwargs needed for the API call
        return {}

class OpenRouterCompletion(Completion):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"
    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)

    def build_kwargs(self, prompt):
        # Return any additional kwargs needed for the API call
        return {}

@llm.hookimpl
def register_models(register):
    # Only do this if the openrouter key is set
    key = llm.get_key("", "openrouter", "OPENROUTER_KEY")
    if not key:
        return
    models = get_openrouter_models()
    for model_definition in models:
        supports_images = get_supports_images(model_definition)
        kwargs = dict(
            model_id="openrouter/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            vision=supports_images,
            supports_schema=has_parameter(model_definition, "structured_outputs"),
            supports_tools=has_parameter(model_definition, "tools"),
            api_base="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(
            OpenRouterChat(**kwargs),
            OpenRouterAsyncChat(**kwargs),
        )

        completion_model = OpenRouterCompletion(
            model_id="openroutercompletion/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            api_base="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(completion_model)

class DownloadError(Exception):
    pass

def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    # Create directories if not exist
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        # Get the file's modification time
        mod_time = path.stat().st_mtime
        # Check if it's more than the cache_timeout old
        if time.time() - mod_time < cache_timeout:
            # If not, load the file
            with open(path, "r") as file:
                return json.load(file)
    # Try to download the data
    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()  # This will raise an HTTPError if the request fails
        # If successful, write to the file
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        # If there's an existing file, load it
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            # If not, raise an error
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def openrouter():
        "Commands relating to the llm-openrouter plugin"

    @openrouter.command()
    @click.option("--free", is_flag=True, help="List free models")
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(free, json_):
        "List of OpenRouter models"
        if free:
            all_models = [
                model
                for model in get_openrouter_models()
                if model["id"].endswith(":free")
            ]
        else:
            all_models = get_openrouter_models()
        if json_:
            click.echo(json.dumps(all_models, indent=2))
        else:
            # Custom format
            for model in all_models:
                bits = []
                bits.append(f"- id: {model['id']}")
                bits.append(f"  name: {model['name']}")
                bits.append(f"  context_length: {model['context_length']:,}")
                architecture = model.get("architecture", None)
                if architecture:
                    bits.append("  architecture:")
                    for key, value in architecture.items():
                        bits.append(
                            "    "
                            + key
                            + ": "
                            + (value if isinstance(value, str) else json.dumps(value))
                        )
                bits.append(f"  supports_schema: {has_parameter(model, 'structured_outputs')}")
                bits.append(f"  supports_tools: {has_parameter(model, 'tools')}")
                pricing = format_pricing(model["pricing"])
                if pricing:
                    bits.append("  pricing: " + pricing)
                click.echo("\n".join(bits) + "\n")

    @openrouter.command()
    @click.option("--key", help="Key to inspect")
    def key(key):
        "View information and rate limits for the current key"
        key = llm.get_key(key, "openrouter", "OPENROUTER_KEY")
        response = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json()["data"], indent=2))


def format_price(key, price_str):
    """Format a price value with appropriate scaling and no trailing zeros."""
    price = float(price_str)

    if price == 0:
        return None

    # Determine scale based on magnitude
    if price < 0.0001:
        scale = 1000000
        suffix = "/M"
    elif price < 0.001:
        scale = 1000
        suffix = "/K"
    elif price < 1:
        scale = 1000
        suffix = "/K"
    else:
        scale = 1
        suffix = ""

    # Scale the price
    scaled_price = price * scale

    # Format without trailing zeros
    # Convert to string and remove trailing .0
    price_str = (
        f"{scaled_price:.10f}".rstrip("0").rstrip(".")
        if "." in f"{scaled_price:.10f}"
        else f"{scaled_price:.0f}"
    )

    return f"{key} ${price_str}{suffix}"


def format_pricing(pricing_dict):
    formatted_parts = []
    for key, value in pricing_dict.items():
        formatted_price = format_price(key, value)
        if formatted_price:
            formatted_parts.append(formatted_price)
    return ", ".join(formatted_parts)
