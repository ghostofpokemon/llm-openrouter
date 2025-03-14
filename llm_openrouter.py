import llm
from llm.default_plugins.openai_models import Chat, Completion
from pathlib import Path
import json
import time
import httpx
from typing import Optional
from pydantic import Field

def get_openrouter_models():
    return fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=3600,
    )["data"]

def load_prefill(name: str) -> Optional[str]:
    """Load a prefill from the prefills directory"""
    path = llm.user_dir() / "prefills" / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    return None

class OpenRouterChat(Chat):
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
    key = llm.get_key("", "openrouter", "LLM_OPENROUTER_KEY")
    if not key:
        return

    models = get_openrouter_models()
    for model_definition in models:
        chat_model = OpenRouterChat(
            model_id="openrouter/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            api_base="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(chat_model)

    for model_definition in models:
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
