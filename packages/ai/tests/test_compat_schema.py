"""Mirror of pi ai test/compat-schema.test.ts.

pi's "preserves property types in the provider superset" case is a TypeScript
type assertion (`expectTypeOf`) with nothing to observe at runtime; it is not
mirrored. The defaults cases read what TypeBox's `Default({})` fills in: the
`default` annotation of each top-level property.
"""

from pidrei_ai.providers.compat_schema import (
    ANTHROPIC_MESSAGES_COMPAT_SCHEMA,
    BEDROCK_COMPAT_SCHEMA,
    OPENAI_COMPLETIONS_COMPAT_SCHEMA,
    OPENAI_RESPONSES_COMPAT_SCHEMA,
    PROVIDER_COMPAT_SCHEMA,
)


def _defaults(schema: dict) -> dict:
    return {name: prop["default"] for name, prop in schema["properties"].items() if "default" in prop}


def test_preserves_api_specific_defaults():
    completions_defaults = _defaults(OPENAI_COMPLETIONS_COMPAT_SCHEMA)
    assert "thinkingFormat" not in completions_defaults
    assert "supportsLongCacheRetention" not in completions_defaults
    assert _defaults(OPENAI_RESPONSES_COMPAT_SCHEMA)["supportsDeveloperRole"] is True
    anthropic_defaults = _defaults(ANTHROPIC_MESSAGES_COMPAT_SCHEMA)
    assert "sendSessionAffinityHeaders" not in anthropic_defaults
    assert anthropic_defaults["supportsLongCacheRetention"] is True
    assert _defaults(BEDROCK_COMPAT_SCHEMA)["supportsStrictMode"] is False


def test_does_not_assign_api_specific_defaults_to_the_provider_superset():
    defaults = _defaults(PROVIDER_COMPAT_SCHEMA)
    assert "supportsDeveloperRole" not in defaults
    assert "sendSessionAffinityHeaders" not in defaults
    assert "supportsLongCacheRetention" not in defaults
    assert "supportsStrictMode" not in defaults
