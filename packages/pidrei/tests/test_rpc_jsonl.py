"""Mirrors pi coding-agent test/rpc-jsonl.test.ts.

pi drives attachJsonlLineReader over a Node Readable; the pidrei decoder is
fed the same chunks directly.
"""

import json

from pidrei.modes.rpc.jsonl import JsonlLineDecoder, serialize_json_line


class TestRpcJsonlFraming:
    def test_serializes_strict_jsonl_records_without_escaping_unicode_separators(self):
        line = serialize_json_line({"text": "a b c"})

        assert "a b c" in line
        assert line.endswith("\n")
        assert json.loads(line.strip()) == {"text": "a b c"}

    def test_splits_on_lf_only_and_preserves_u2028_u2029_inside_payloads(self):
        decoder = JsonlLineDecoder()

        lines = decoder.feed(serialize_json_line({"text": "a b c"}))
        lines.extend(decoder.end())

        assert len(lines) == 1
        assert json.loads(lines[0]) == {"text": "a b c"}

    def test_handles_crlf_delimited_input(self):
        decoder = JsonlLineDecoder()

        lines = decoder.feed(b'{"a":1}\r\n{"b":2}\r\n')
        lines.extend(decoder.end())

        assert lines == ['{"a":1}', '{"b":2}']

    def test_emits_a_final_line_without_trailing_lf(self):
        decoder = JsonlLineDecoder()

        lines = decoder.feed(b'{"a":1}')
        lines.extend(decoder.end())

        assert lines == ['{"a":1}']

    def test_reassembles_lines_split_across_chunks_and_multibyte_boundaries(self):
        # Not in pi's suite (Node's StringDecoder covers it there); guards the
        # incremental utf-8 decoder against mid-codepoint chunk splits.
        payload = serialize_json_line({"text": "héllo"}).encode("utf-8")
        decoder = JsonlLineDecoder()

        lines: list[str] = []
        for index in range(len(payload)):
            lines.extend(decoder.feed(payload[index : index + 1]))
        lines.extend(decoder.end())

        assert len(lines) == 1
        assert json.loads(lines[0]) == {"text": "héllo"}
