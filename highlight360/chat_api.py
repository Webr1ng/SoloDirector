"""Small, non-retrying OpenAI-compatible chat transport with safe diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from contextlib import suppress
from urllib.parse import urlsplit

import requests


MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_METADATA_CHARS = 256
_MAX_COUNT = (1 << 63) - 1
_AUDIT_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", re.ASCII)
_PROJECT_ENV = Path(__file__).resolve().parent.parent / ".env"
_PROJECT_CREDENTIALS = {"OPENAI_API_KEY", "OPENAI_BASE_URL",
                        "DASHSCOPE_API_KEY", "DASHSCOPE_HTTP_BASE_URL"}


class APIRequestError(RuntimeError):
    """A fixed, public-safe message; ``code`` is an optional provider audit ID.

    HTTP failures use a validated provider error code/type, or ``HTTP`` when
    unavailable. The provider's message and original exception are never shown.
    Optional metadata is sanitized by the transport; no response text is retained.
    """

    def __init__(self, code: str, status_code: int | None = None, *, metadata: dict | None = None):
        self.status_code = status_code
        self.code = code
        self.metadata = {key: metadata[key]
                         for key in ("usage", "latency_sec", "request_id", "response_model")
                         if metadata is not None and key in metadata}
        if status_code is not None and not 200 <= status_code < 300:
            message = f"Chat API HTTP {status_code}: request rejected."
        else:
            message = {
                "timeout": "Chat API request timed out; it was not retried.",
                "connection_error": "Chat API connection failed; it was not retried.",
                "invalid_response": "Chat API returned an invalid or oversized response.",
                "truncated": "Chat API response was truncated.",
            }.get(code, "Chat API request failed.")
        super().__init__(message)


def _validate_url(value: str, *, proxy: bool = False) -> str:
    # Check before urlsplit: it silently removes some control characters.
    message = ("H360_HTTP_PROXY must be an explicit HTTP/HTTPS proxy URL."
               if proxy else
               "API base URL must be explicit HTTPS without userinfo, query or fragment.")
    if (not isinstance(value, str) or not value
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)
            or "\\" in value or "?" in value or "#" in value):
        raise ValueError(message)
    try:
        parsed = urlsplit(value)
        allowed = ("http", "https") if proxy else ("https",)
        if (parsed.scheme not in allowed or not parsed.netloc or not parsed.hostname
                or "%" in parsed.hostname
                or (not proxy and (parsed.username is not None or parsed.password is not None))
                or (proxy and parsed.path not in ("", "/"))
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)
                or parsed.netloc.endswith(":")):
            raise ValueError
    except ValueError:
        raise ValueError(message) from None
    return value.rstrip("/")


def _validate_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An OPENAI_API_KEY or DASHSCOPE_API_KEY is required.")
    key = value.strip()
    if any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError("API key contains unsupported header characters.")
    return key


def _env_value(name: str) -> str:
    value = os.environ.get(name, "")
    return value if value.strip() else ""


def _read_project_env() -> dict[str, str]:
    """Read only credential settings from the ignored project .env; never execute it."""
    try:
        if _PROJECT_ENV.stat().st_size > 16 * 1024:
            raise ValueError("Project .env is unexpectedly large.")
        lines = _PROJECT_ENV.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        raise ValueError("Project .env cannot be read.") from None
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() in _PROJECT_CREDENTIALS:
            values[name.strip()] = value.strip()
    return values


def resolve_api_environment() -> tuple[str, str]:
    """Resolve process values first, then explicit project .env credentials.

    A DashScope REST base ending in /api/v1 is converted in place, preserving
    its host, region and any custom workspace prefix. An OpenAI base is used as-is.
    """
    project = _read_project_env()

    key = _validate_key(_env_value("OPENAI_API_KEY") or _env_value("DASHSCOPE_API_KEY")
                        or project.get("OPENAI_API_KEY", "") or project.get("DASHSCOPE_API_KEY", ""))
    env_openai = _env_value("OPENAI_BASE_URL")
    if env_openai:
        return _validate_url(env_openai), key
    env_dashscope = _env_value("DASHSCOPE_HTTP_BASE_URL")
    project_openai = project.get("OPENAI_BASE_URL", "")
    if project_openai and not env_dashscope:
        return _validate_url(project_openai), key
    base = env_dashscope or project.get("DASHSCOPE_HTTP_BASE_URL", "")
    if not base:
        raise ValueError("OPENAI_BASE_URL or DASHSCOPE_HTTP_BASE_URL is required.")
    base = _validate_url(base)
    if urlsplit(base).path.endswith("/api/v1"):
        base = base[:-len("/api/v1")] + "/compatible-mode/v1"
    return base, key


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant.")


def _decode_json(raw: bytes, status_code: int) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        if not isinstance(payload, dict):
            raise ValueError
    except (ValueError, RecursionError):
        raise APIRequestError("invalid_response", status_code) from None
    return payload


def _read_response(response: requests.Response, status_code: int) -> bytes:
    length = response.headers.get("Content-Length", "")
    if isinstance(length, str) and length.isascii() and length.isdigit():
        # Avoid converting an attacker-controlled, arbitrarily long integer.
        normalized = length.lstrip("0") or "0"
        if len(normalized) > 10 or int(normalized) > MAX_RESPONSE_BYTES:
            raise APIRequestError("invalid_response", status_code)
    raw = bytearray()
    # stream=True bounds the decompressed bytes too, not just Content-Length.
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        if not isinstance(chunk, bytes) or len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
            raise APIRequestError("invalid_response", status_code)
        raw.extend(chunk)
    return bytes(raw)


def _safe_usage(value: object) -> dict:
    if not isinstance(value, dict):
        return {}

    def counts(source: dict, names: tuple[str, ...]) -> dict:
        return {name: source[name] for name in names
                if type(source.get(name)) is int and 0 <= source[name] <= _MAX_COUNT}

    result = counts(value, ("prompt_tokens", "completion_tokens", "total_tokens"))
    for name, fields in (
        ("prompt_tokens_details", ("cached_tokens", "cache_read_tokens", "cache_creation_tokens")),
        ("completion_tokens_details", ("reasoning_tokens",)),
    ):
        details = value.get(name)
        if isinstance(details, dict):
            safe = counts(details, fields)
            if safe:
                result[name] = safe
    return result


def _safe_metadata(value: object, api_key: str) -> str:
    if (not isinstance(value, str) or api_key in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        return ""
    return value.strip()[:MAX_METADATA_CHARS]


def _http_error_code(payload: dict, api_key: str, content: str | list) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        for field in ("code", "type"):
            value = error.get(field)
            if (isinstance(value, str) and _AUDIT_CODE.fullmatch(value)
                    and api_key not in value
                    and not (isinstance(content, str) and content.strip() in value)):
                return value
    return "HTTP"


class ChatClient:
    """Reusable client; every complete() owns and closes a separate Session.

    No model-specific thinking flags are injected. Session's default adapters
    have retries disabled, including for billable POST requests.
    """

    def __init__(self, base_url: str, api_key: str, timeout_sec: int = 120):
        self.base_url = _validate_url(base_url)
        self._api_key = _validate_key(api_key)
        if type(timeout_sec) is not int or timeout_sec <= 0:
            raise ValueError("timeout_sec must be a positive integer.")
        # This is a network read timeout, not a generation/token limit.
        self.timeout_sec = timeout_sec

    def complete(self, model: str, content: str | list) -> dict:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string.")
        if (not isinstance(content, (str, list)) or not content
                or (isinstance(content, str) and not content.strip())):
            raise ValueError("content must be a nonempty string or list.")
        proxy = _env_value("H360_HTTP_PROXY")
        if proxy:
            proxy = _validate_url(proxy, proxy=True)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }
        session = None
        response = None
        status_code = None
        started = time.perf_counter()
        try:
            session = requests.Session()
            session.trust_env = False
            session.proxies.clear()
            if proxy:
                session.proxies.update({"http": proxy, "https": proxy})
            response = session.post(
                self.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + self._api_key},
                json=body, timeout=(10, self.timeout_sec),
                verify=True, allow_redirects=False, stream=True,
            )
            status = response.status_code
            if type(status) is not int or not 100 <= status <= 599:
                raise APIRequestError("invalid_response")
            status_code = status
            if not 200 <= status < 300:
                code = "HTTP"
                try:
                    payload = _decode_json(_read_response(response, status), status)
                    code = _http_error_code(payload, self._api_key, content)
                except APIRequestError:
                    pass  # Preserve the HTTP failure even for HTML/oversized errors.
                raise APIRequestError(code, status)
            payload = _decode_json(_read_response(response, status), status)
            # A rejected/truncated answer still consumed tokens. Extract only safe
            # billing/audit fields before validating choices, never the raw payload.
            request_id = ""
            for candidate in (response.headers.get("x-request-id"),
                              response.headers.get("x-dashscope-request-id"),
                              payload.get("request_id"), payload.get("id")):
                request_id = _safe_metadata(candidate, self._api_key)
                if request_id:
                    break
            metadata = {
                "usage": _safe_usage(payload.get("usage")),
                "latency_sec": float(time.perf_counter() - started),
                "response_model": _safe_metadata(payload.get("model"), self._api_key),
                "request_id": request_id,
            }
            choices = payload.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise APIRequestError("invalid_response", status, metadata=metadata)
            choice = choices[0]
            if choice.get("finish_reason") == "length":
                raise APIRequestError("truncated", status, metadata=metadata)
            if choice.get("finish_reason") != "stop":
                raise APIRequestError("invalid_response", status, metadata=metadata)
            message = choice.get("message")
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise APIRequestError("invalid_response", status, metadata=metadata)
            return {"text": text, **metadata}
        except APIRequestError:
            raise
        except requests.exceptions.Timeout:
            raise APIRequestError("timeout", status_code) from None
        except requests.exceptions.RequestException:
            raise APIRequestError("connection_error", status_code) from None
        except (ValueError, TypeError, RecursionError):
            # Serialization/decoding errors can otherwise contain private input.
            raise APIRequestError("invalid_response", status_code) from None
        finally:
            # Cleanup must not replace a sanitized error with a raw exception.
            if response is not None:
                with suppress(Exception):
                    response.close()
            if session is not None:
                with suppress(Exception):
                    session.close()
