"""Port of pi's azure-openai-config.ts: the Azure endpoint and deployment
helpers shared by the Responses adapter and the Azure provider.

pi reads the `azure*` options off whatever options object arrives. Options are
dataclasses here, so the helpers read those fields from an
`AzureEndpointOptions` only; any other `StreamOptions` has none.
"""

from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from pidrei_ai.types import Model, StreamOptions
from pidrei_ai.utils.provider_env import get_provider_env_value


DEFAULT_AZURE_API_VERSION = "v1"

_AZURE_HOST_SUFFIXES = (".openai.azure.com", ".cognitiveservices.azure.com", ".ai.azure.com")
_AZURE_ROOT_PATHS = ("", "/", "/openai", "/openai/v1/responses")


@dataclass(slots=True)
class AzureEndpointOptions(StreamOptions):
    """Azure models ship without a base_url: one resource per user, resolved per request."""

    azure_api_version: str | None = None
    azure_resource_name: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None


def _endpoint_options(options: StreamOptions | None) -> AzureEndpointOptions | None:
    return options if isinstance(options, AzureEndpointOptions) else None


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value:
        return result
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parts = trimmed.split("=", 1)
        if len(parts) != 2:
            continue
        model_id, deployment_name = parts
        if not model_id or not deployment_name:
            continue
        result[model_id.strip()] = deployment_name.strip()
    return result


def resolve_deployment_name(model: Model, options: StreamOptions | None = None) -> str:
    azure = _endpoint_options(options)
    if azure is not None and azure.azure_deployment_name:
        return azure.azure_deployment_name
    mapped = _parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.env if options else None)
    ).get(model.id)
    return mapped or model.id


def normalize_azure_base_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}")

    hostname = (parsed.hostname or "").lower()
    is_azure_host = hostname.endswith(_AZURE_HOST_SUFFIXES)
    normalized_path = parsed.path.rstrip("/")

    # Azure hosts need /openai/v1 as the base path so `/deployments/<model>/...`
    # and `?api-version=v1` append correctly.
    if is_azure_host and normalized_path in _AZURE_ROOT_PATHS:
        parsed = parsed._replace(path="/openai/v1", query="")

    return urlunparse(parsed).rstrip("/")


def build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def resolve_azure_base_url(model: Model, options: StreamOptions | None = None) -> str:
    azure = _endpoint_options(options)
    env = options.env if options else None
    base_url = (azure.azure_base_url.strip() if azure is not None and azure.azure_base_url else None) or (
        (get_provider_env_value("AZURE_OPENAI_BASE_URL", env) or "").strip() or None
    )
    resource_name = (azure.azure_resource_name if azure is not None else None) or get_provider_env_value(
        "AZURE_OPENAI_RESOURCE_NAME", env
    )

    resolved_base_url = base_url
    if not resolved_base_url and resource_name:
        resolved_base_url = build_default_base_url(resource_name)
    if not resolved_base_url and model.base_url:
        resolved_base_url = model.base_url
    if not resolved_base_url:
        raise RuntimeError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, "
            "or pass azure_base_url, azure_resource_name, or model.base_url."
        )

    return normalize_azure_base_url(resolved_base_url)


def resolve_azure_config(model: Model, options: StreamOptions | None = None) -> tuple[str, str]:
    """Returns `(base_url, api_version)`."""
    azure = _endpoint_options(options)
    return (
        resolve_azure_base_url(model, options),
        (azure.azure_api_version if azure is not None else None)
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", options.env if options else None)
        or DEFAULT_AZURE_API_VERSION,
    )
