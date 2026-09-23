"""Offline transport contract tests. All keys, endpoints and responses are fake."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from threading import Barrier
from unittest.mock import Mock, patch

import requests
from requests.structures import CaseInsensitiveDict

from highlight360 import chat_api
from highlight360.chat_api import APIRequestError, ChatClient, resolve_api_environment


BASE = "https://workspace.example.invalid/team-a/compatible-mode/v1"
KEY = "offline-secret-key-987654"
PROMPT = "private prompt: do not include in diagnostics"


def completion(**changes):
    value = {
        "id": "chatcmpl-offline",
        "model": "qwen-vl-max",
        "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"score": 9}',
            "reasoning_content": "private reasoning must not become the answer",
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    value.update(changes)
    return value


def fake_response(payload=None, *, status=200, raw=None, headers=None):
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = CaseInsensitiveDict(headers or {})
    if raw is None:
        raw = json.dumps(completion() if payload is None else payload,
                         ensure_ascii=False).encode("utf-8")
    response.iter_content.return_value = [raw]
    return response


def fake_session(response=None):
    session = Mock(spec=["post", "close", "proxies", "trust_env", "mount"])
    session.trust_env = True
    session.proxies = {"https": "socks5://ignored.example.invalid:1080"}
    session.post.return_value = response if response is not None else fake_response()
    return session


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        # Replace the mapping instead of inspecting/copying actual environment keys.
        self.start_patch(patch.object(os, "environ", {}))
        self.start_patch(patch.object(chat_api, "_read_project_env", return_value={}))
        # Belt-and-suspenders guard: even a broken Session mock cannot use the wire.
        self.start_patch(patch.object(requests.sessions.Session, "request",
                                      side_effect=AssertionError("Network is forbidden in tests.")))
        self.session = fake_session()
        self.factory = self.start_patch(patch.object(chat_api.requests, "Session",
                                                     return_value=self.session))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def assert_safe(self, error):
        displayed = "\n".join((str(error), repr(error), repr(error.__dict__),
                                 "".join(traceback.format_exception(
                                     type(error), error, error.__traceback__))))
        for private in (KEY, PROMPT, "password-in-url", "raw-response-marker"):
            self.assertNotIn(private, displayed)


class ProjectEnvTests(unittest.TestCase):
    def test_project_env_is_read_without_executing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text(
                "# local credentials\nOPENAI_API_KEY=project-fake-key\n"
                "OPENAI_BASE_URL=https://project.example.invalid/v1\n"
                "UNRELATED_SETTING=ignored\n", encoding="utf-8")
            with patch.object(chat_api, "_PROJECT_ENV", env_file), patch.object(os, "environ", {}):
                self.assertEqual(resolve_api_environment(),
                                 ("https://project.example.invalid/v1", "project-fake-key"))
                self.assertNotIn("UNRELATED_SETTING", chat_api._read_project_env())
                os.environ.update({"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": BASE})
                self.assertEqual(resolve_api_environment(), (BASE, KEY))


class EnvironmentTests(OfflineTestCase):
    def test_openai_values_take_priority_independently(self):
        os.environ.update({
            "OPENAI_API_KEY": KEY,
            "DASHSCOPE_API_KEY": "other-offline-key",
            "OPENAI_BASE_URL": BASE + "/",
            "DASHSCOPE_HTTP_BASE_URL": "https://other.example.invalid/api/v1",
        })
        self.assertEqual(resolve_api_environment(), (BASE, KEY))
        self.factory.assert_not_called()

    def test_dashscope_fallback_preserves_custom_workspace_and_port(self):
        os.environ.update({
            "DASHSCOPE_API_KEY": KEY,
            "DASHSCOPE_HTTP_BASE_URL": "https://regional.example.invalid:9443/workspaces/team-a/api/v1/",
        })
        self.assertEqual(resolve_api_environment(), (
            "https://regional.example.invalid:9443/workspaces/team-a/compatible-mode/v1", KEY))

    def test_key_and_base_selection_are_independent(self):
        for environment in (
            {"OPENAI_API_KEY": KEY, "DASHSCOPE_HTTP_BASE_URL": BASE},
            {"DASHSCOPE_API_KEY": KEY, "OPENAI_BASE_URL": BASE},
        ):
            with self.subTest(environment=list(environment)):
                os.environ.clear()
                os.environ.update(environment)
                self.assertEqual(resolve_api_environment(), (BASE, KEY))

    def test_empty_preferred_values_fall_back(self):
        for empty in ("", " \t\n"):
            with self.subTest(empty=empty):
                os.environ.update({"OPENAI_API_KEY": empty, "OPENAI_BASE_URL": empty,
                                   "DASHSCOPE_API_KEY": KEY, "DASHSCOPE_HTTP_BASE_URL": BASE})
                self.assertEqual(resolve_api_environment(), (BASE, KEY))

    def test_missing_or_blank_key(self):
        os.environ["OPENAI_BASE_URL"] = BASE
        for value in (None, "", " \t"):
            with self.subTest(value=value):
                if value is not None:
                    os.environ["OPENAI_API_KEY"] = value
                with self.assertRaisesRegex(ValueError, "API_KEY") as caught:
                    resolve_api_environment()
                self.assert_safe(caught.exception)
        self.factory.assert_not_called()

    def test_missing_base_never_guesses_public_address(self):
        for key_name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
            with self.subTest(key_name=key_name):
                os.environ.clear()
                os.environ[key_name] = KEY
                with self.assertRaisesRegex(ValueError, "BASE_URL") as caught:
                    resolve_api_environment()
                self.assert_safe(caught.exception)
        self.factory.assert_not_called()

    def test_invalid_preferred_base_does_not_fall_back(self):
        os.environ.update({"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": "http://unsafe.example.invalid",
                           "DASHSCOPE_HTTP_BASE_URL": BASE})
        with self.assertRaises(ValueError):
            resolve_api_environment()

    def test_invalid_preferred_key_does_not_fall_back(self):
        os.environ.update({"OPENAI_API_KEY": KEY + "\r\ninjected", "DASHSCOPE_API_KEY": "safe-fake-key",
                           "OPENAI_BASE_URL": BASE})
        with self.assertRaises(ValueError) as caught:
            resolve_api_environment()
        self.assert_safe(caught.exception)

    def test_conversion_only_changes_dashscope_path_suffix(self):
        cases = (
            ("https://api/v1", "https://api/v1"),
            ("https://example.invalid/api/v1/nested", "https://example.invalid/api/v1/nested"),
            ("https://example.invalid/team/api/v1", "https://example.invalid/team/compatible-mode/v1"),
            (BASE, BASE),
        )
        os.environ["DASHSCOPE_API_KEY"] = KEY
        for source, expected in cases:
            with self.subTest(source=source):
                os.environ["DASHSCOPE_HTTP_BASE_URL"] = source
                self.assertEqual(resolve_api_environment(), (expected, KEY))
        os.environ["OPENAI_BASE_URL"] = "https://example.invalid/custom/api/v1"
        self.assertEqual(resolve_api_environment()[0], os.environ["OPENAI_BASE_URL"])

    def test_base_rejects_non_https_and_ambiguous_or_secret_urls(self):
        bad_urls = (
            "http://example.invalid/v1", "ftp://example.invalid", "//example.invalid/v1",
            "example.invalid/v1", "https:///v1", "https://", "https://:443/v1",
            "https://user:password-in-url@example.invalid/v1", "https://user@example.invalid",
            "https://@example.invalid/v1", BASE + "?key=" + KEY, BASE + "#" + KEY,
            BASE + "?", BASE + "#", "https://example.invalid\\@other.invalid",
            "https://example.invalid:wrong/v1", "https://example.invalid:65536/v1",
            "https://example.invalid:0/v1", "https://example.invalid:/v1",
            "https://[broken/v1", "https://exam%70le.invalid/v1",
            "https://example.invalid/space here", "https://example.invalid/\nprivate",
            "https://example.invalid/\tprivate", "\r" + BASE, BASE + "\x00",
        )
        os.environ["OPENAI_API_KEY"] = KEY
        for source in bad_urls:
            with self.subTest(source=source):
                os.environ["OPENAI_BASE_URL"] = source
                with self.assertRaises(ValueError) as caught:
                    resolve_api_environment()
                self.assert_safe(caught.exception)
                with self.assertRaises(ValueError) as caught:
                    ChatClient(source, KEY)
                self.assert_safe(caught.exception)
        self.factory.assert_not_called()

    def test_invalid_dashscope_base_is_checked_before_conversion(self):
        os.environ.update({"DASHSCOPE_API_KEY": KEY,
                           "DASHSCOPE_HTTP_BASE_URL": "http://example.invalid/api/v1"})
        with self.assertRaises(ValueError):
            resolve_api_environment()

    def test_explicit_https_ipv6_and_custom_paths_are_allowed(self):
        for base in ("https://[::1]:8443/workspace/api/v1", BASE, "https://example.invalid"):
            with self.subTest(base=base):
                self.assertEqual(ChatClient(base + "/", KEY).base_url, base)


class RequestTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.client = ChatClient(BASE + "///", KEY)

    def test_exact_default_request_contract(self):
        with patch.object(chat_api.time, "perf_counter", side_effect=[10.0, 10.75]):
            result = self.client.complete("qwen-vl-max", PROMPT)
        self.factory.assert_called_once_with()
        self.session.post.assert_called_once_with(
            BASE + "/chat/completions",
            headers={"Authorization": "Bearer " + KEY},
            json={"model": "qwen-vl-max", "messages": [{"role": "user", "content": PROMPT}],
                  "response_format": {"type": "json_object"}},
            timeout=(10, 120), verify=True, allow_redirects=False, stream=True,
        )
        self.assertEqual(result, {
            "text": '{"score": 9}',
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            "latency_sec": 0.75, "response_model": "qwen-vl-max", "request_id": "chatcmpl-offline",
        })
        self.assertIsInstance(result["latency_sec"], float)
        self.session.post.return_value.iter_content.assert_called_once_with(chunk_size=64 * 1024)
        self.session.post.return_value.json.assert_not_called()
        self.session.post.return_value.close.assert_called_once_with()
        self.session.close.assert_called_once_with()
        self.session.mount.assert_not_called()
        self.assertNotIn(KEY, repr(self.client))

    def test_multiframe_content_is_passed_without_thinking_override(self):
        content = [{"type": "text", "text": "Compare both frames as JSON."},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZTE="}},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZTI="}}]
        self.client.complete("qwen-vl-max", content)
        body = self.session.post.call_args.kwargs["json"]
        self.assertIs(body["messages"][0]["content"], content)
        self.assertEqual(set(body), {"model", "messages", "response_format"})
        self.assertEqual(len(body["messages"]), 1)
        self.assertNotIn("enable_thinking", body)
        self.assertNotIn("thinking", body)
        self.assertNotIn("extra_body", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_custom_read_timeout_is_not_a_token_limit(self):
        ChatClient(BASE, KEY, timeout_sec=45).complete("configured-model", "text")
        kwargs = self.session.post.call_args.kwargs
        self.assertEqual(kwargs["timeout"], (10, 45))
        self.assertEqual(set(kwargs["json"]), {"model", "messages", "response_format"})
        self.assertFalse(hasattr(self.client, "max_tokens"))
        for name in ("max_tokens", "max_completion_tokens"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                ChatClient(BASE, KEY, **{name: 0})

    def test_all_five_models_use_json_object_without_token_or_thinking_overrides(self):
        from highlight360.experts import EXPERTS
        for expert in EXPERTS:
            with self.subTest(model=expert["model"]):
                self.client.complete(expert["model"], PROMPT)
                body = self.session.post.call_args.kwargs["json"]
                self.assertEqual(body, {"model": expert["model"],
                                       "messages": [{"role": "user", "content": PROMPT}],
                                       "response_format": {"type": "json_object"}})
        self.assertEqual(self.session.post.call_count, 5)

    def test_invalid_constructor_options_never_start_session(self):
        for value in (None, True, False, 0, -1, "120", 1.5, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ChatClient(BASE, KEY, timeout_sec=value)
        for key in (None, "", "  ", KEY + "\nheader", KEY + "\x00", KEY + "非ASCII"):
            with self.subTest(key_type=type(key)), self.assertRaises(ValueError) as caught:
                ChatClient(BASE, key)
            self.assert_safe(caught.exception)
        self.factory.assert_not_called()

    def test_invalid_inputs_never_start_session(self):
        for model in (None, True, 1, [], "", "  "):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.client.complete(model, PROMPT)
        for content in (None, True, 1, {}, (), [], "", " \n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.client.complete("model", content)
        self.factory.assert_not_called()

    def test_system_socks_and_certificate_environment_are_ignored(self):
        os.environ.update({
            "HTTP_PROXY": "socks5://system.invalid:1080", "HTTPS_PROXY": "socks5://system.invalid:1080",
            "ALL_PROXY": "socks5h://system.invalid:1080", "http_proxy": "socks5://system.invalid:1080",
            "https_proxy": "socks5://system.invalid:1080", "all_proxy": "socks5://system.invalid:1080",
            "NO_PROXY": "*", "REQUESTS_CA_BUNDLE": "not-a-real-certificate",
            "CURL_CA_BUNDLE": "not-a-real-certificate", "NETRC": "not-a-real-netrc",
        })
        self.client.complete("model", PROMPT)
        self.assertIs(self.session.trust_env, False)
        self.assertEqual(self.session.proxies, {})
        self.assertIs(self.session.post.call_args.kwargs["verify"], True)

    def test_only_explicit_http_or_https_proxy_is_used(self):
        for proxy in ("http://127.0.0.1:8080", "https://proxy.example.invalid:8443",
                      "http://user:password-in-url@proxy.example.invalid:8080"):
            with self.subTest(proxy=proxy):
                os.environ.update({"H360_HTTP_PROXY": proxy, "ALL_PROXY": "socks5://ignored.invalid:1"})
                self.client.complete("model", PROMPT)
                self.assertIs(self.session.trust_env, False)
                self.assertEqual(self.session.proxies, {"http": proxy, "https": proxy})

    def test_blank_explicit_proxy_means_direct_connection(self):
        os.environ["H360_HTTP_PROXY"] = " \t"
        self.client.complete("model", PROMPT)
        self.assertEqual(self.session.proxies, {})

    def test_invalid_explicit_proxy_rejected_without_request(self):
        for proxy in ("socks5://localhost:1080", "proxy.invalid:8080", "http:///missing",
                      "http://proxy.invalid/path", "http://proxy.invalid?key=" + KEY,
                      "http://proxy.invalid#" + KEY, "http://proxy.invalid:70000",
                      "http://proxy.invalid/\n"):
            with self.subTest(proxy=proxy):
                os.environ["H360_HTTP_PROXY"] = proxy
                with self.assertRaises(ValueError) as caught:
                    self.client.complete("model", PROMPT)
                self.assert_safe(caught.exception)
        self.factory.assert_not_called()

    def test_each_call_owns_a_fresh_session(self):
        sessions = [fake_session(), fake_session()]
        self.factory.side_effect = sessions
        for _ in range(2):
            self.client.complete("model", PROMPT)
        self.assertEqual(self.factory.call_count, 2)
        for session in sessions:
            session.post.assert_called_once()
            session.close.assert_called_once_with()
            session.post.return_value.close.assert_called_once_with()

    def test_same_client_supports_overlapping_thread_calls(self):
        gate = Barrier(2, timeout=5)
        sessions = [fake_session(), fake_session()]
        responses = [fake_response(completion(id="thread-one")), fake_response(completion(id="thread-two"))]

        def receive(response):
            def post(*args, **kwargs):
                gate.wait()
                return response
            return post

        for session, response in zip(sessions, responses):
            session.post.side_effect = receive(response)
        self.factory.side_effect = sessions
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: self.client.complete("model", PROMPT), range(2)))
        self.assertEqual({result["request_id"] for result in results}, {"thread-one", "thread-two"})
        self.assertEqual(self.factory.call_count, 2)
        for session, response in zip(sessions, responses):
            session.post.assert_called_once()
            session.close.assert_called_once()
            response.close.assert_called_once()


class ClientTestCase(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.client = ChatClient(BASE, KEY)

    def invoke(self, response):
        self.session.post.return_value = response
        return self.client.complete("model", PROMPT)

    def assert_failure(self, response, code="invalid_response", status=200):
        with self.assertRaises(APIRequestError) as caught:
            self.invoke(response)
        error = caught.exception
        self.assertEqual(error.code, code)
        self.assertEqual(error.status_code, status)
        self.assertIsInstance(error, RuntimeError)
        self.assert_safe(error)
        response.close.assert_called_once_with()
        return error


class ResponseTests(ClientTestCase):
    def test_success_utf8_chunked_and_preserves_exact_text(self):
        data = completion()
        data["choices"][0]["message"]["content"] = '  {"理由":"完整互动"}\n'
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        response = fake_response()
        response.iter_content.return_value = [b""] + [raw[i:i + 1] for i in range(len(raw))]
        result = self.invoke(response)
        self.assertEqual(result["text"], data["choices"][0]["message"]["content"])

    def test_transport_does_not_reinterpret_answer_as_reasoning_or_parse_task_schema(self):
        data = completion()
        data["choices"][0]["message"]["content"] = "model answer"
        self.assertEqual(self.invoke(fake_response(data))["text"], "model answer")

    def test_rejects_malformed_non_object_and_non_strict_json(self):
        bad = (b"", b" ", b"not JSON", b"<html>raw-response-marker</html>", b"\xff",
               b"[]", b"null", b"true", b"123", b'"text"', b"{} trailing", b"{}{}",
               b'{"choices": [], "choices": []}', b'{"extra":NaN}', b'{"extra":Infinity}',
               b'{"extra":-Infinity}', b'{"choices":[],}', b"[" * 1500 + b"]" * 1500)
        for raw in bad:
            with self.subTest(raw=raw[:40]):
                self.assert_failure(fake_response(raw=raw))

    def test_requires_exactly_one_object_choice(self):
        for choices in (None, [], {}, "text", [None], [1], ["text"], [{}], [{}, {}]):
            with self.subTest(choices=choices):
                self.assert_failure(fake_response(completion(choices=choices)))
        data = completion()
        del data["choices"]
        self.assert_failure(fake_response(data))

    def test_length_finish_is_truncated_even_with_answer(self):
        for text in ('{"score": 9}', "", None):
            data = completion()
            data["choices"][0]["finish_reason"] = "length"
            data["choices"][0]["message"]["content"] = text
            with self.subTest(text=text):
                error = self.assert_failure(fake_response(data), "truncated")
                self.assertEqual(error.metadata["usage"], data["usage"])
                self.assertEqual(error.metadata["request_id"], "chatcmpl-offline")

    def test_truncated_response_retains_only_sanitized_usage_and_audit_fields(self):
        data = completion(usage={
            "prompt_tokens": 20, "completion_tokens": 4096, "total_tokens": 4116,
            "prompt_tokens_details": {"cached_tokens": 12, "private": KEY},
            "completion_tokens_details": {"reasoning_tokens": 3000, "raw": PROMPT},
            "credentials": KEY, "raw": "raw-response-marker",
        })
        data["choices"][0]["finish_reason"] = "length"
        data["choices"][0]["message"].update(content=KEY + PROMPT + "raw-response-marker",
                                             reasoning_content="raw-response-marker")
        data["private"] = PROMPT
        response = fake_response(data, headers={"x-request-id": "truncated-request-id"})
        with patch.object(chat_api.time, "perf_counter", side_effect=[10.0, 12.5]):
            error = self.assert_failure(response, "truncated")
        self.assertEqual(error.metadata, {
            "usage": {"prompt_tokens": 20, "completion_tokens": 4096, "total_tokens": 4116,
                      "prompt_tokens_details": {"cached_tokens": 12},
                      "completion_tokens_details": {"reasoning_tokens": 3000}},
            "latency_sec": 2.5, "request_id": "truncated-request-id", "response_model": "qwen-vl-max",
        })
        self.assertNotIn("text", error.metadata)
        self.assertNotIn("raw", error.__dict__)
        self.assert_safe(error)

    def test_truncated_metadata_sanitizes_keys_and_invalid_count_types(self):
        data = completion(model=KEY, id="unsafe\nline", usage={
            "prompt_tokens": True, "completion_tokens": -1, "total_tokens": "raw-response-marker",
            "completion_tokens_details": {"reasoning_tokens": 3, "private": KEY},
        })
        data["choices"][0]["finish_reason"] = "length"
        error = self.assert_failure(fake_response(data, headers={"x-request-id": KEY}), "truncated")
        self.assertEqual(error.metadata["usage"], {"completion_tokens_details": {"reasoning_tokens": 3}})
        self.assertEqual(error.metadata["request_id"], "")
        self.assertEqual(error.metadata["response_model"], "")
        self.assertIsInstance(error.metadata["latency_sec"], float)

    def test_invalid_choice_after_safe_json_parse_also_retains_consumption(self):
        error = self.assert_failure(fake_response(completion(choices=[])))
        self.assertEqual(error.metadata["usage"]["total_tokens"], 28)
        self.assertEqual(error.metadata["request_id"], "chatcmpl-offline")

    def test_all_other_non_stop_finishes_fail(self):
        for finish in (None, "", "tool_calls", "function_call", "content_filter", "error", True, [], {}):
            data = completion()
            data["choices"][0]["finish_reason"] = finish
            with self.subTest(finish=finish):
                self.assert_failure(fake_response(data))
        data = completion()
        del data["choices"][0]["finish_reason"]
        self.assert_failure(fake_response(data))

    def test_requires_nonempty_string_content_never_reasoning_fallback(self):
        for content in (None, "", " \r\n", [], [{"text": "answer"}], {}, 1, True):
            data = completion()
            data["choices"][0]["message"]["content"] = content
            with self.subTest(content=content):
                self.assert_failure(fake_response(data))
        data = completion()
        del data["choices"][0]["message"]["content"]
        self.assert_failure(fake_response(data))
        for message in (None, "text", [], True):
            data = completion()
            data["choices"][0]["message"] = message
            with self.subTest(message=message):
                self.assert_failure(fake_response(data))

    def test_rejects_missing_or_invalid_status(self):
        for status in (None, "200", True, 99, 600, 200.0):
            with self.subTest(status=status):
                self.assert_failure(fake_response(status=status), status=None)

    def test_empty_204_is_not_successful_answer(self):
        self.assert_failure(fake_response(status=204, raw=b""), status=204)

    def test_response_limit_with_and_without_content_length(self):
        with patch.object(chat_api, "MAX_RESPONSE_BYTES", 64):
            for headers in ({}, {"Content-Length": "65"}, {"Content-Length": "1"},
                            {"Content-Length": "9" * 5000}, {"Content-Length": "invalid"}):
                with self.subTest(header_present=bool(headers)):
                    self.assert_failure(fake_response(raw=b"x" * 65, headers=headers))

    def test_oversized_advertised_length_stops_before_body(self):
        response = fake_response(headers={"Content-Length": str(chat_api.MAX_RESPONSE_BYTES + 1)})
        self.assert_failure(response)
        response.iter_content.assert_not_called()

    def test_stream_limit_stops_reading_and_closes(self):
        def chunks():
            yield b"x" * 32
            yield b"x" * 33
            raise AssertionError("Must not drain oversized response.")

        response = fake_response()
        response.iter_content.return_value = chunks()
        with patch.object(chat_api, "MAX_RESPONSE_BYTES", 64):
            self.assert_failure(response)

    def test_exact_response_limit_is_allowed(self):
        raw = json.dumps(completion()).encode("utf-8")
        with patch.object(chat_api, "MAX_RESPONSE_BYTES", len(raw)):
            self.assertEqual(self.invoke(fake_response(raw=raw, headers={
                "Content-Length": str(len(raw))}))["text"], '{"score": 9}')

    def test_usage_whitelist_keeps_safe_counts_only(self):
        usage = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17,
                 "prompt_tokens_details": {"cached_tokens": 10, "cache_read_tokens": 2,
                                           "cache_creation_tokens": 0, "secret": KEY, "audio_tokens": 5},
                 "completion_tokens_details": {"reasoning_tokens": 3, "private": PROMPT},
                 "private": KEY, "cost": 0.3, "other": {"nested": PROMPT}}
        actual = self.invoke(fake_response(completion(usage=usage)))["usage"]
        self.assertEqual(actual, {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17,
                                 "prompt_tokens_details": {"cached_tokens": 10, "cache_read_tokens": 2,
                                                           "cache_creation_tokens": 0},
                                 "completion_tokens_details": {"reasoning_tokens": 3}})
        self.assertNotIn(KEY, repr(actual))
        self.assertNotIn(PROMPT, repr(actual))

    def test_bad_usage_types_are_ignored(self):
        for value in (None, True, "12", [], 1, -1, 1.5):
            with self.subTest(value=value):
                self.assertEqual(self.invoke(fake_response(completion(usage=value)))["usage"], {})
        for value in (None, True, False, "12", [], {}, -1, 1.5, 1 << 80):
            usage = {"prompt_tokens": value, "completion_tokens": value, "total_tokens": value,
                     "prompt_tokens_details": {"cached_tokens": value},
                     "completion_tokens_details": {"reasoning_tokens": value}}
            with self.subTest(count=value):
                self.assertEqual(self.invoke(fake_response(completion(usage=usage)))["usage"], {})
        for details in (None, True, 1, [], "secret"):
            usage = {"total_tokens": 0, "prompt_tokens_details": details,
                     "completion_tokens_details": details}
            with self.subTest(details=details):
                self.assertEqual(self.invoke(fake_response(completion(usage=usage)))["usage"],
                                 {"total_tokens": 0})

    def test_absent_optional_fields_have_safe_empty_defaults(self):
        data = completion()
        for key in ("usage", "model", "id"):
            del data[key]
        result = self.invoke(fake_response(data))
        self.assertEqual(result["usage"], {})
        self.assertEqual(result["response_model"], "")
        self.assertEqual(result["request_id"], "")

    def test_metadata_lengths_and_bad_types(self):
        data = completion(model="m" * 1000, id="i" * 1000)
        result = self.invoke(fake_response(data))
        self.assertEqual(len(result["response_model"]), chat_api.MAX_METADATA_CHARS)
        self.assertEqual(len(result["request_id"]), chat_api.MAX_METADATA_CHARS)
        for value in (None, 3, True, {}, [], KEY, "unsafe\nline", "unsafe\x00line"):
            with self.subTest(value_type=type(value)):
                result = self.invoke(fake_response(completion(model=value, id=value)))
                self.assertEqual(result["response_model"], "")
                self.assertEqual(result["request_id"], "")

    def test_request_id_sources_and_case_insensitive_headers(self):
        for headers, body, expected in (
            ({"X-Request-ID": "header-id", "X-DashScope-Request-ID": "dash-id"},
             {"request_id": "body-id"}, "header-id"),
            ({"X-DashScope-Request-ID": "dash-id"}, {"request_id": "body-id"}, "dash-id"),
            ({}, {"request_id": "body-id"}, "body-id"),
            ({"X-Request-ID": KEY}, {}, "chatcmpl-offline"),
        ):
            with self.subTest(expected=expected):
                result = self.invoke(fake_response(completion(**body), headers=headers))
                self.assertEqual(result["request_id"], expected)
        result = self.invoke(fake_response(headers={"X-Request-ID": "h" * 1000}))
        self.assertEqual(len(result["request_id"]), chat_api.MAX_METADATA_CHARS)


class ErrorTests(ClientTestCase):
    def test_http_failures_and_redirects_never_retry_or_follow_location(self):
        for status in (301, 302, 307, 308, 400, 401, 403, 404, 408, 429, 500, 503):
            with self.subTest(status=status):
                self.factory.reset_mock()
                self.session.reset_mock()
                response = fake_response({"error": {"code": "ProductNotActivated",
                                                       "message": KEY + PROMPT + "raw-response-marker"}},
                                         status=status, headers={"Location": "https://other.invalid?key=" + KEY})
                self.assert_failure(response, "ProductNotActivated", status)
                self.factory.assert_called_once()
                self.session.post.assert_called_once()
                self.session.close.assert_called_once()
                self.assertIs(self.session.post.call_args.kwargs["allow_redirects"], False)

    def test_http_audit_code_whitelist_and_type_fallback(self):
        for error, expected in (
            ({"code": "rate_limit_exceeded"}, "rate_limit_exceeded"),
            ({"code": "AccessDenied.Unpurchased"}, "AccessDenied.Unpurchased"),
            ({"code": "429"}, "429"),
            ({"code": "bad code " + KEY, "type": "invalid_request_error"}, "invalid_request_error"),
            ({"code": KEY, "type": "unauthorized"}, "unauthorized"),
            ({"code": "x" * 65}, "HTTP"),
            ({"code": 401, "type": []}, "HTTP"),
            ({"code": "错误"}, "HTTP"),
            ({"code": "https://other.invalid?key=" + KEY}, "HTTP"),
            ({"code": "bad\ncode"}, "HTTP"),
            ({"code": None, "type": "auth-error"}, "auth-error"),
            ({"message": "productnotactivated " + KEY + PROMPT}, "HTTP"),
            (None, "HTTP"), ([], "HTTP"), ("raw-response-marker", "HTTP"),
        ):
            with self.subTest(expected=expected):
                self.assert_failure(fake_response({"error": error}, status=403), expected, 403)

    def test_short_prompt_echo_cannot_be_audit_code(self):
        self.session.post.return_value = fake_response({"error": {"code": "privateprompt"}}, status=400)
        with self.assertRaises(APIRequestError) as caught:
            self.client.complete("model", "privateprompt")
        self.assertEqual(caught.exception.code, "HTTP")
        self.assertNotIn("privateprompt", str(caught.exception))

    def test_non_json_and_oversized_http_failures_preserve_status(self):
        for raw in (b"", b"<html>raw-response-marker</html>", b"null", b"x" * 129):
            with self.subTest(size=len(raw)), patch.object(chat_api, "MAX_RESPONSE_BYTES", 128):
                self.assert_failure(fake_response(raw=raw, status=502), "HTTP", 502)

    def test_transport_errors_are_sanitized_closed_and_not_retried(self):
        private = KEY + " " + PROMPT + " https://server.invalid?password-in-url raw-response-marker"
        for error_type, expected in (
            (requests.exceptions.Timeout, "timeout"),
            (requests.exceptions.ConnectTimeout, "timeout"),
            (requests.exceptions.ReadTimeout, "timeout"),
            (requests.exceptions.ConnectionError, "connection_error"),
            (requests.exceptions.SSLError, "connection_error"),
            (requests.exceptions.ProxyError, "connection_error"),
            (requests.exceptions.RequestException, "connection_error"),
        ):
            with self.subTest(error_type=error_type.__name__):
                self.factory.reset_mock()
                self.session.reset_mock()
                self.session.post.side_effect = error_type(private)
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(APIRequestError) as caught:
                        self.client.complete("model", PROMPT)
                self.assertEqual(caught.exception.code, expected)
                self.assertIsNone(caught.exception.status_code)
                self.assertEqual(caught.exception.metadata, {})
                self.assertTrue(caught.exception.__suppress_context__)
                self.assert_safe(caught.exception)
                self.assertEqual(output.getvalue(), "")
                self.factory.assert_called_once()
                self.session.post.assert_called_once()
                self.session.close.assert_called_once()

    def test_stream_failures_close_response_and_session(self):
        for error, code in ((requests.exceptions.ReadTimeout(KEY), "timeout"),
                            (requests.exceptions.ChunkedEncodingError(KEY), "connection_error")):
            self.session.reset_mock()
            response = fake_response()
            response.iter_content.side_effect = error
            with self.subTest(code=code):
                caught = self.assert_failure(response, code)
                self.assertTrue(caught.__suppress_context__)
                self.session.post.assert_called_once()
                self.session.close.assert_called_once()

    def test_serialization_failures_do_not_echo_input(self):
        for error in (ValueError(KEY + PROMPT), TypeError(KEY + PROMPT)):
            self.session.post.side_effect = error
            with self.subTest(error_type=type(error)), self.assertRaises(APIRequestError) as caught:
                self.client.complete("model", PROMPT)
            self.assertEqual(caught.exception.code, "invalid_response")
            self.assert_safe(caught.exception)

    def test_cleanup_errors_never_leak_or_prevent_session_close(self):
        response = fake_response(raw=b"invalid JSON")
        response.close.side_effect = RuntimeError(KEY + "raw-response-marker")
        self.session.close.side_effect = RuntimeError(KEY + PROMPT)
        self.assert_failure(response)
        self.session.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
