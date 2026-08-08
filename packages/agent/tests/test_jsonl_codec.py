"""JSONL v4 codec round-trips (mirror of pi agent/test/harness/session/jsonl-codec.test.ts)."""

import json

import pytest

from pidrei_agent.harness.session.jsonl.codec import (
    encode_header,
    encode_mutation,
    metadata_from_header,
    parse_header,
    parse_mutation,
)
from pidrei_agent.harness.session.jsonl.types import JsonlSessionMetadata, JsonlV4Header
from pidrei_agent.harness.session.state import (
    EntryMutation,
    LabelFactMutation,
    LaneMutation,
    NameFactMutation,
    RecordMutation,
    SessionMutation,
)
from pidrei_agent.harness.session.types import CustomEntry, OperationStartedRecord, RunIntent, SessionError


def expect_header_round_trip(header: JsonlV4Header) -> None:
    encoded = encode_header(header)
    assert encoded.endswith("\n")
    assert parse_header(encoded.rstrip("\n"), "/sessions/example.jsonl") == header


def expect_mutation_round_trip(mutation: SessionMutation) -> None:
    encoded = encode_mutation(mutation)
    assert encoded.endswith("\n")
    assert parse_mutation(encoded.rstrip("\n"), "/sessions/example.jsonl", 2) == mutation


def test_round_trips_every_header_field_with_a_resolved_parent():
    expect_header_round_trip(
        JsonlV4Header(
            id="session",
            created_at=1_700_000_000_000,
            cwd="/workspace/project",
            parent_session_id="parent",
            metadata={"owner": "agent", "nested": {"enabled": True}, "values": [1, None, "two"]},
        )
    )


def test_round_trips_an_unresolved_legacy_parent_path():
    expect_header_round_trip(
        JsonlV4Header(
            id="legacy-child",
            created_at=1_700_000_000_001,
            cwd="/workspace/project",
            legacy_parent_session_path="/sessions/missing-parent.jsonl",
        )
    )


def test_projects_header_and_filesystem_fields_into_metadata():
    header = JsonlV4Header(
        id="session",
        created_at=1_700_000_000_000,
        cwd="/workspace/project",
        legacy_parent_session_path="/sessions/missing-parent.jsonl",
        metadata={"owner": "agent"},
    )

    assert metadata_from_header(header, "/sessions/session.jsonl", 1_700_000_000_100) == JsonlSessionMetadata(
        id="session",
        created_at=1_700_000_000_000,
        cwd="/workspace/project",
        path="/sessions/session.jsonl",
        modified_at=1_700_000_000_100,
        source_format=4,
        legacy_parent_session_path="/sessions/missing-parent.jsonl",
        metadata={"owner": "agent"},
    )


def test_round_trips_a_lane_bound_entry_line():
    expect_mutation_round_trip(
        EntryMutation(
            lane="main",
            entry=CustomEntry(
                id="entry-1", seq=1, parent_id=None, timestamp=100, custom_type="note", data={"text": "hello"}
            ),
        )
    )


def test_round_trips_an_imported_entry_line_without_a_lane():
    expect_mutation_round_trip(
        EntryMutation(entry=CustomEntry(id="entry-1", seq=1, parent_id=None, timestamp=100, custom_type="note"))
    )


def test_round_trips_a_record_line():
    expect_mutation_round_trip(
        RecordMutation(
            record=OperationStartedRecord(
                id="run-1",
                seq=1,
                lane="main",
                timestamp=100,
                source_leaf_id=None,
                intent=RunIntent(original_prompt=[], initial_messages=[]),
            )
        )
    )


def test_round_trips_a_lane_line():
    expect_mutation_round_trip(LaneMutation(seq=1, lane="thread", leaf_id="entry-1"))


def test_round_trips_both_fact_line_discriminants():
    expect_mutation_round_trip(NameFactMutation(seq=1, name="Example"))
    expect_mutation_round_trip(LabelFactMutation(seq=2, target_id="entry-1", label="checkpoint"))


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        (
            "a custom entry without customType",
            {"kind": "entry", "type": "custom", "id": "entry", "parentId": None, "seq": 1, "timestamp": 1},
        ),
        (
            "an operation_started record without intent",
            {
                "kind": "record",
                "type": "operation_started",
                "id": "run",
                "lane": "main",
                "seq": 1,
                "timestamp": 1,
                "sourceLeafId": None,
            },
        ),
        (
            "an operation_finished record without runId",
            {
                "kind": "record",
                "type": "operation_finished",
                "id": "finish",
                "lane": "main",
                "seq": 1,
                "timestamp": 1,
                "outcome": "completed",
            },
        ),
    ],
)
def test_rejects_invalid_mutation_lines(name, mutation):
    with pytest.raises(SessionError):
        parse_mutation(json.dumps(mutation), "/sessions/example.jsonl", 2)
