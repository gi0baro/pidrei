"""Port of pi's openai-decisions.lazy.ts."""

from pidrei_ai.types import ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, ProviderClassifier


class _LazyOpenAIDecisionsApi(ProviderClassifier):
    async def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> ClassifierResult:
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import openai_decisions

        return await openai_decisions.classify(model, context, options)


def openai_decisions_api() -> ProviderClassifier:
    return _LazyOpenAIDecisionsApi()
