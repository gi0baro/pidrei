"""Port of pi's cloudflare-workers-ai-system-one.lazy.ts."""

from pidrei_ai.types import ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, ProviderClassifier


class _LazyCloudflareWorkersAISystemOneApi(ProviderClassifier):
    async def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> ClassifierResult:
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import cloudflare_workers_ai_system_one

        return await cloudflare_workers_ai_system_one.classify(model, context, options)


def cloudflare_workers_ai_system_one_api() -> ProviderClassifier:
    return _LazyCloudflareWorkersAISystemOneApi()
