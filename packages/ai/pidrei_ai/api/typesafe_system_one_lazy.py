"""Port of pi's typesafe-system-one.lazy.ts."""

from pidrei_ai.types import ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, ProviderClassifier


class _LazyTypeSafeSystemOneApi(ProviderClassifier):
    async def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> ClassifierResult:
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import typesafe_system_one

        return await typesafe_system_one.classify(model, context, options)


def typesafe_system_one_api() -> ProviderClassifier:
    return _LazyTypeSafeSystemOneApi()
