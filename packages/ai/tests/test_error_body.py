"""Mirror of pi's error-body.test.ts (backfilled during the 0.82.1 sync — the
production module shipped without its mirrored tests).

pi synthesizes one error per JS-SDK shape; the Python port probes the
snake_case fields our adapters and punkreq raise (`status_code`, `status`,
`body`, `error` — see error_body.py's docstring), so the two Bedrock
`$metadata`/`$response` cases (including the unread-response-stream guard from
pi af3b934f) have no counterpart here by design.
"""

import json

from pidrei_ai.utils.error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    format_provider_error,
    normalize_provider_error,
)


def _error(message: str, **attributes) -> Exception:
    error = Exception(message)
    for name, value in attributes.items():
        setattr(error, name, value)
    return error


class TestNormalizeProviderError:
    def test_extracts_status_and_body_from_a_mistral_shaped_error(self):
        error = _error("Mistral request failed", status_code=403, body='{"error":"blocked by gateway WAF"}')

        norm = normalize_provider_error(error)

        assert norm.status == 403
        assert norm.body == '{"error":"blocked by gateway WAF"}'
        assert norm.message_carries_body is False

    def test_reads_the_parsed_body_off_an_api_error_when_the_message_is_opaque(self):
        # The openai-style client yields "<status> status code (no body)" when
        # the parsed body is unparsed, while the body stays on `error.error`.
        error = _error("403 status code (no body)", status=403, error={"error": "blocked by gateway WAF"})

        norm = normalize_provider_error(error)

        assert norm.status == 403
        assert norm.body == '{"error":"blocked by gateway WAF"}'
        assert norm.message_carries_body is False

    def test_preserves_the_message_when_the_client_already_folds_the_body_into_it(self):
        body = json.dumps({"error": {"code": 403, "message": "Permission denied"}}, separators=(",", ":"))
        error = _error(body, status=403)

        norm = normalize_provider_error(error)

        assert norm.status == 403
        assert norm.message_carries_body is True
        assert norm.message == body

    def test_json_stringifies_a_non_error_thrown_value(self):
        norm = normalize_provider_error({"reason": "boom"})

        assert norm.status is None
        assert norm.body is None
        assert norm.message == '{"reason":"boom"}'
        assert norm.message_carries_body is False

    def test_treats_an_empty_parsed_body_object_as_no_body(self):
        error = _error("403 status code (no body)", status=403, error={})

        norm = normalize_provider_error(error)

        assert norm.body is None
        assert norm.message_carries_body is True

    def test_truncates_the_body_at_the_cap(self):
        long_body = "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 50)
        error = _error("failed", status_code=500, body=long_body)

        norm = normalize_provider_error(error)

        assert "... [truncated 50 chars]" in norm.body
        assert len(norm.body) < len(long_body)

    def test_sets_message_carries_body_when_the_message_already_contains_the_extracted_body(self):
        error = _error("500: upstream exploded", status_code=500, body="upstream exploded")

        norm = normalize_provider_error(error)

        assert norm.message_carries_body is True


class TestFormatProviderError:
    def test_surfaces_status_and_body_without_a_prefix(self):
        norm = normalize_provider_error(
            _error("403 status code (no body)", status=403, error={"error": "blocked by gateway WAF"})
        )

        formatted = format_provider_error(norm)

        assert "403" in formatted
        assert "blocked by gateway WAF" in formatted
        assert formatted != "403 status code (no body)"

    def test_applies_a_provider_prefix_with_status_and_body(self):
        norm = normalize_provider_error(
            _error("403 status code (no body)", status=403, error={"error": "blocked by gateway WAF"})
        )

        assert (
            format_provider_error(norm, "OpenAI API error")
            == 'OpenAI API error (403): {"error":"blocked by gateway WAF"}'
        )

    def test_preserves_the_message_with_prefix_and_status_when_it_already_carries_the_body(self):
        body = json.dumps({"error": {"message": "Permission denied"}}, separators=(",", ":"))
        norm = normalize_provider_error(_error(body, status=403))

        assert format_provider_error(norm, "OpenAI API error") == f"OpenAI API error (403): {body}"

    def test_returns_the_bare_message_for_a_non_error_value(self):
        norm = normalize_provider_error({"reason": "boom"})

        assert format_provider_error(norm) == '{"reason":"boom"}'
