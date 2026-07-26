"""Port of pi's bedrock-provider.ts: the eagerly-importable adapter module."""

from pidrei_ai.api.bedrock_converse_stream import stream, stream_simple


bedrock_provider_module = {"stream": stream, "stream_simple": stream_simple}
