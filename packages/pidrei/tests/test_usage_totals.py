"""Mirror of pi coding-agent test/usage-totals.test.ts."""

from pidrei.core.usage_totals import combine_usage
from pidrei_ai.types import Usage, UsageCost


def usage(total: float, total_from_provider: bool | None = None) -> Usage:
    return Usage(
        input=1,
        output=1,
        total_tokens=2,
        cost=UsageCost(total=total, total_from_provider=total_from_provider),
    )


class TestCombineUsage:
    def test_marks_the_combined_total_as_provider_reported_only_when_both_totals_are(self):
        combined = combine_usage(usage(1, True), usage(2, True)).cost
        assert combined.total == 3
        assert combined.total_from_provider is True
        assert combine_usage(usage(1, True), usage(2)).cost.total_from_provider is False
        assert combine_usage(usage(1), usage(2, True)).cost.total_from_provider is False
        assert combine_usage(usage(1), usage(2)).cost.total_from_provider is False
