"""Tests for Gmail body_format support across helper and public tool APIs."""

import base64
from unittest.mock import Mock

import pytest

import gmail.gmail_tools as gmail_tools
from core.utils import UserInputError
from gmail.gmail_helpers import DEFAULT_GMAIL_BODY_MAX_CHARS, get_gmail_body_max_chars
from gmail.gmail_tools import (
    _extract_message_bodies,
    _format_body_content,
    get_gmail_message_content,
    get_gmail_messages_content_batch,
    get_gmail_thread_content,
    get_gmail_threads_content_batch,
)


def _unwrap(tool):
    """Unwrap FunctionTool + decorators to the original async function."""
    fn = tool.fn if hasattr(tool, "fn") else tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _encode(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def _headers(**overrides):
    header_map = {
        "Subject": "Example subject",
        "From": "sender@example.com",
        "To": "recipient@example.com",
        "Cc": "cc@example.com",
        "Message-ID": "<message@example.com>",
        "Date": "Fri, 28 Mar 2026 10:00:00 -0400",
    }
    header_map.update(overrides)
    return [{"name": name, "value": value} for name, value in header_map.items()]


def _payload(headers=None, text=None, html=None):
    payload = {"headers": headers or _headers()}
    parts = []
    if text is not None:
        parts.append({"mimeType": "text/plain", "body": {"data": _encode(text)}})
    if html is not None:
        parts.append({"mimeType": "text/html", "body": {"data": _encode(html)}})
    if parts:
        payload["mimeType"] = "multipart/alternative"
        payload["parts"] = parts
    return payload


def _message_response(message_id: str, text="", html="", headers=None):
    return {
        "id": message_id,
        "payload": _payload(headers=headers, text=text, html=html),
    }


def _metadata_response(message_id: str, headers=None):
    return {
        "id": message_id,
        "payload": {"headers": headers or _headers()},
    }


def _thread_message(message_id: str, text="", html="", headers=None):
    return {
        "id": message_id,
        "payload": _payload(headers=headers, text=text, html=html),
    }


class _FakeBatch:
    def __init__(self, callback):
        self._callback = callback
        self._requests = []

    def add(self, request, request_id):
        self._requests.append((request_id, request))

    def execute(self):
        for request_id, request in self._requests:
            try:
                response = request.execute()
                self._callback(request_id, response, None)
            except Exception as exc:
                self._callback(request_id, None, exc)


def _build_service(*, message_responses=None, thread_responses=None):
    message_responses = message_responses or {}
    thread_responses = thread_responses or {}

    service = Mock()

    def message_get(**kwargs):
        request = Mock()
        response = message_responses[(kwargs["id"], kwargs["format"])]
        if isinstance(response, Exception):
            request.execute.side_effect = response
        else:
            request.execute.return_value = response
        return request

    def thread_get(**kwargs):
        request = Mock()
        response = thread_responses[(kwargs["id"], kwargs["format"])]
        if isinstance(response, Exception):
            request.execute.side_effect = response
        else:
            request.execute.return_value = response
        return request

    service.users().messages().get.side_effect = message_get
    service.users().threads().get.side_effect = thread_get
    service.new_batch_http_request.side_effect = lambda callback: _FakeBatch(callback)
    return service


class TestFormatBodyContentTextMode:
    """Verify default 'text' body_format preserves existing behavior."""

    def test_returns_text_body_when_available(self):
        result = _format_body_content("Hello world", "<b>Hello world</b>")
        assert result == "Hello world"

    def test_returns_text_body_default_format(self):
        result = _format_body_content(
            "Hello world", "<b>Hello world</b>", body_format="text"
        )
        assert result == "Hello world"

    def test_falls_back_to_html_when_text_empty(self):
        result = _format_body_content("", "<p>HTML content here</p>")
        assert "HTML content here" in result

    def test_returns_no_content_when_both_empty(self):
        result = _format_body_content("", "")
        assert result == "[No readable content found]"

    def test_detects_low_value_placeholder_text(self):
        low_value = "Your client does not support HTML messages"
        html = "<p>This is the actual email content with much more detail</p>"
        result = _format_body_content(low_value, html)
        assert "actual email content" in result

    def test_truncates_long_html_fallback(self):
        long_html = "<p>" + "x" * 25000 + "</p>"
        result = _format_body_content("", long_html)
        assert "[Content truncated...]" in result


class TestFormatBodyContentHtmlMode:
    """Verify 'html' body_format returns raw HTML."""

    def test_returns_raw_html_body(self):
        html = "<div><b>Hello</b> <em>world</em></div>"
        result = _format_body_content("Hello world", html, body_format="html")
        assert result == html

    def test_returns_html_without_conversion(self):
        html = "<table><tr><td>Cell</td></tr></table>"
        result = _format_body_content("Cell", html, body_format="html")
        assert "<table>" in result
        assert "<td>Cell</td>" in result

    def test_falls_back_to_text_when_no_html(self):
        result = _format_body_content("Plain text only", "", body_format="html")
        assert result == "Plain text only"

    def test_returns_no_content_when_both_empty(self):
        result = _format_body_content("", "", body_format="html")
        assert result == "[No readable content found]"

    def test_strips_whitespace_from_html(self):
        result = _format_body_content("text", "  <b>html</b>  ", body_format="html")
        assert result == "<b>html</b>"

    def test_truncates_long_html(self):
        long_html = "<div>" + "x" * 25000 + "</div>"
        result = _format_body_content("text", long_html, body_format="html")
        assert "[Content truncated...]" in result
        assert len(result) < len(long_html)

    def test_preserves_html_entities(self):
        html = "<p>Price: &lt;$100 &amp; free shipping</p>"
        result = _format_body_content("", html, body_format="html")
        assert "&lt;" in result
        assert "&amp;" in result

    def test_preserves_style_and_script_tags(self):
        html = "<style>body { color: red; }</style><p>Content</p>"
        result = _format_body_content("Content", html, body_format="html")
        assert "<style>" in result
        assert "color: red" in result

    def test_whitespace_only_html_falls_back_to_text(self):
        result = _format_body_content("Fallback text", "   \n\t  ", body_format="html")
        assert result == "Fallback text"


class TestGmailBodyMaxChars:
    """WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS governs every body truncation site."""

    MARKER = "\n\n[Content truncated...]"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, DEFAULT_GMAIL_BODY_MAX_CHARS),
            ("", DEFAULT_GMAIL_BODY_MAX_CHARS),
            ("0", None),
            (" 50000 ", 50000),
        ],
    )
    def test_parse_rules(self, monkeypatch, raw, expected):
        """Unset keeps the historical 20,000-character cap; 0 disables truncation."""
        if raw is None:
            monkeypatch.delenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", raising=False)
        else:
            monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", raw)
        assert DEFAULT_GMAIL_BODY_MAX_CHARS == 20000
        assert get_gmail_body_max_chars() == expected

    @pytest.mark.parametrize("raw", ["-1", "ten", "1.5", "20k"])
    def test_rejects_invalid(self, monkeypatch, raw):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", raw)
        with pytest.raises(ValueError, match="WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS"):
            get_gmail_body_max_chars()

    def test_default_truncates_at_20000_characters(self, monkeypatch):
        monkeypatch.delenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", raising=False)
        html = "x" * 25000
        result = _format_body_content("text", html, body_format="html")
        assert result == "x" * 20000 + self.MARKER

    def test_html_format_honours_configured_limit(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "10")
        result = _format_body_content("text", "<b>" + "x" * 50 + "</b>", "html")
        assert result == "<b>xxxxxxx" + self.MARKER

    def test_html_format_zero_disables_truncation(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "0")
        html = "<div>" + "x" * 25000 + "</div>"
        result = _format_body_content("text", html, body_format="html")
        assert result == html

    def test_text_from_html_fallback_honours_configured_limit(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "10")
        result = _format_body_content("", "<p>" + "y" * 50 + "</p>")
        assert result == "y" * 10 + self.MARKER

    def test_text_from_html_fallback_zero_disables_truncation(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "0")
        result = _format_body_content("", "<p>" + "y" * 25000 + "</p>")
        assert result == "y" * 25000

    def test_raw_mime_honours_configured_limit(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "10")
        result = gmail_tools._decode_raw_mime_content(_encode("z" * 50))
        assert result == "z" * 10 + self.MARKER

    def test_raw_mime_zero_disables_truncation(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "0")
        result = gmail_tools._decode_raw_mime_content(_encode("z" * 25000))
        assert result == "z" * 25000


class TestExtractMessageBodies:
    """Verify _extract_message_bodies extracts both text and HTML parts."""

    def test_extracts_text_and_html_from_multipart(self):
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _encode("Plain text")},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _encode("<b>HTML</b>")},
                },
            ],
        }
        bodies = _extract_message_bodies(payload)
        assert bodies["text"] == "Plain text"
        assert bodies["html"] == "<b>HTML</b>"

    def test_extracts_text_only(self):
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _encode("Just text")},
        }
        bodies = _extract_message_bodies(payload)
        assert bodies["text"] == "Just text"
        assert bodies["html"] == ""

    def test_extracts_html_only(self):
        payload = {
            "mimeType": "text/html",
            "body": {"data": _encode("<p>Just HTML</p>")},
        }
        bodies = _extract_message_bodies(payload)
        assert bodies["text"] == ""
        assert bodies["html"] == "<p>Just HTML</p>"

    def test_handles_empty_payload(self):
        bodies = _extract_message_bodies({})
        assert bodies["text"] == ""
        assert bodies["html"] == ""

    def test_handles_nested_multipart(self):
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": _encode("Nested text")},
                        },
                        {
                            "mimeType": "text/html",
                            "body": {"data": _encode("<p>Nested HTML</p>")},
                        },
                    ],
                },
            ],
        }
        bodies = _extract_message_bodies(payload)
        assert bodies["text"] == "Nested text"
        assert bodies["html"] == "<p>Nested HTML</p>"


@pytest.mark.asyncio
async def test_get_gmail_message_content_returns_raw_mime():
    service = _build_service(
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
            ("msg-1", "raw"): {"raw": _encode("Raw MIME body")},
        }
    )

    result = await _unwrap(get_gmail_message_content)(
        service=service,
        message_id="msg-1",
        user_google_email="user@example.com",
        body_format="raw",
    )

    assert "--- RAW MIME ---" in result
    assert "Raw MIME body" in result
    assert "From: sender@example.com" in result
    assert "Date: Fri, 28 Mar 2026 10:00:00 -0400" in result
    assert "To: recipient@example.com" in result
    assert "Cc: cc@example.com" in result
    assert "From:    " not in result


@pytest.mark.asyncio
async def test_get_gmail_message_content_reports_raw_decode_errors():
    service = _build_service(
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
            ("msg-1", "raw"): {"raw": "a"},
        }
    )

    result = await _unwrap(get_gmail_message_content)(
        service=service,
        message_id="msg-1",
        user_google_email="user@example.com",
        body_format="raw",
    )

    assert "[Failed to decode raw MIME:" in result


@pytest.mark.asyncio
async def test_get_gmail_message_content_truncates_raw_mime(monkeypatch):
    monkeypatch.setenv("WORKSPACE_MCP_GMAIL_BODY_MAX_CHARS", "12")
    service = _build_service(
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
            ("msg-1", "raw"): {"raw": _encode("x" * 32)},
        }
    )

    result = await _unwrap(get_gmail_message_content)(
        service=service,
        message_id="msg-1",
        user_google_email="user@example.com",
        body_format="raw",
    )

    assert "--- RAW MIME ---" in result
    assert "[Content truncated...]" in result


@pytest.mark.asyncio
async def test_get_gmail_messages_content_batch_supports_raw_format():
    service = _build_service(
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
            ("msg-1", "raw"): {"raw": _encode("Batch raw MIME body")},
        }
    )

    result = await _unwrap(get_gmail_messages_content_batch)(
        service=service,
        message_ids=["msg-1"],
        user_google_email="user@example.com",
        body_format="raw",
    )

    assert "Retrieved 1 messages" in result
    assert "--- RAW MIME ---" in result
    assert "Batch raw MIME body" in result

    formats = [
        call.kwargs["format"]
        for call in service.users.return_value.messages.return_value.get.call_args_list
    ]
    assert formats.count("metadata") == 1
    assert formats.count("raw") == 1


@pytest.mark.asyncio
async def test_get_gmail_messages_content_batch_default_text_format():
    service = _build_service(
        message_responses={
            ("msg-1", "full"): _message_response(
                "msg-1", text="Plain text body", html="<p>HTML body</p>"
            ),
        }
    )

    result = await _unwrap(get_gmail_messages_content_batch)(
        service=service,
        message_ids=["msg-1"],
        user_google_email="user@example.com",
    )

    assert "Plain text body" in result
    assert "--- BODY ---" in result
    assert "--- RAW MIME ---" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("body_format", ["html", "raw"])
async def test_get_gmail_messages_content_batch_rejects_metadata_with_body_format(
    body_format,
):
    service = _build_service(
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
        }
    )

    with pytest.raises(UserInputError, match="require format='full'"):
        await _unwrap(get_gmail_messages_content_batch)(
            service=service,
            message_ids=["msg-1"],
            user_google_email="user@example.com",
            format="metadata",
            body_format=body_format,
        )


@pytest.mark.asyncio
async def test_get_gmail_thread_content_supports_raw_format():
    service = _build_service(
        message_responses={
            ("msg-1", "raw"): {"raw": _encode("Thread raw MIME 1")},
            ("msg-2", "raw"): {"raw": _encode("Thread raw MIME 2")},
        },
        thread_responses={
            (
                "thread-1",
                "full",
            ): {
                "messages": [
                    _thread_message("msg-1", text="Plain 1", html="<p>HTML 1</p>"),
                    _thread_message("msg-2", text="Plain 2", html="<p>HTML 2</p>"),
                ]
            }
        },
    )

    result = await _unwrap(get_gmail_thread_content)(
        service=service,
        thread_id="thread-1",
        user_google_email="user@example.com",
        body_format="raw",
    )

    assert result.count("--- RAW MIME ---") == 2
    assert "Thread raw MIME 1" in result
    assert "Thread raw MIME 2" in result


@pytest.mark.asyncio
async def test_get_gmail_threads_content_batch_supports_raw_format():
    service = _build_service(
        message_responses={
            ("msg-1", "raw"): {"raw": _encode("Batch thread raw MIME")},
        },
        thread_responses={
            (
                "thread-1",
                "full",
            ): {
                "messages": [
                    _thread_message("msg-1", text="Plain 1", html="<p>HTML 1</p>")
                ]
            }
        },
    )

    result = await _unwrap(get_gmail_threads_content_batch)(
        service=service,
        thread_ids=["thread-1"],
        user_google_email="user@example.com",
        body_format="raw",
    )

    assert "Retrieved 1 threads:" in result
    assert "--- RAW MIME ---" in result
    assert "Batch thread raw MIME" in result


@pytest.mark.asyncio
async def test_get_gmail_message_content_preserves_html_format():
    service = _build_service(
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
            (
                "msg-1",
                "full",
            ): _message_response(
                "msg-1", text="Plain fallback", html="<p><b>HTML</b></p>"
            ),
        }
    )

    result = await _unwrap(get_gmail_message_content)(
        service=service,
        message_id="msg-1",
        user_google_email="user@example.com",
        body_format="html",
    )

    assert "<p><b>HTML</b></p>" in result
    assert "From: sender@example.com" in result
    assert "Date: Fri, 28 Mar 2026 10:00:00 -0400" in result
    assert "To: recipient@example.com" in result
    assert "Cc: cc@example.com" in result
    assert "From:    " not in result
