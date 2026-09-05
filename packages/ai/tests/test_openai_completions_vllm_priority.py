"""Mirror of pi's openai-completions-vllm-priority.test.ts.

pi captures the payload through a mocked OpenAI SDK; pidrei builds the request
body directly, which is the same observable object.
"""

from pidrei_ai.api.openai_completions import build_params
from pidrei_ai.types import OpenAICompletionsCompat
from tests.test_openai_completions import make_model, opts, user_context


def test_sends_compat_vllm_priority_as_the_top_level_priority_request_field():
    params = build_params(make_model(compat=OpenAICompletionsCompat(vllm_priority=10)), user_context(), opts())
    assert params["priority"] == 10


def test_omits_priority_when_vllm_priority_is_not_set():
    params = build_params(make_model(), user_context(), opts())
    assert "priority" not in params
