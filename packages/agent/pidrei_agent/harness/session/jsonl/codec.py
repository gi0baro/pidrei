"""JSONL v4 line codec (port of pi `session/jsonl/codec.ts`).

pi persists raw JS objects with `JSON.stringify`, so v4 session lines carry
camelCase keys and omit absent fields. Messages and usage ride the shared serde
converters; entries and records map to the v4 dataclasses with explicit
per-type field maps. Provisioned entries nested inside records (queue targets,
run intents) serialize without `parentId`/`seq`/`timestamp`, exactly like pi's
`ProvisionedEntry`.
"""

import json
from typing import Any

from ...types import Result, err, ok
from ..serde import parse_message, parse_usage, serialize_message, serialize_usage, to_wire_value
from ..state import (
    EntryMutation,
    LabelFactMutation,
    LaneMutation,
    NameFactMutation,
    RecordMutation,
    SessionMutation,
)
from ..types import (
    AbortRequestedRecord,
    ActiveToolsEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CompactionIntent,
    CustomEntry,
    Entry,
    LaneRecord,
    MessageEntry,
    ModelChangeEntry,
    NavigationIntent,
    OperationFinishedRecord,
    OperationIntent,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RunIntent,
    StepAttemptRecord,
    ThinkingLevelEntry,
    ToolStartedRecord,
    UsageRecord,
    WriteDeferredRecord,
)
from .errors import JsonlDecodeError
from .types import JsonlSessionMetadata, JsonlV4Header


ENTRY_TYPES = {
    "message",
    "model_change",
    "thinking_level_change",
    "active_tools_change",
    "compaction",
    "branch_summary",
    "custom",
}
RECORD_TYPES = {
    "operation_started",
    "abort_requested",
    "operation_finished",
    "step_attempt",
    "tool_started",
    "queue_enqueued",
    "queue_cancelled",
    "write_deferred",
    "usage",
}
OPERATION_KINDS = {"run", "compaction", "navigation"}


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _parse_object(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except ValueError as error:
        raise JsonlDecodeError("syntax", "is not valid JSON", error) from error
    if not _is_object(value):
        raise JsonlDecodeError("schema", "is not a JSON object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise JsonlDecodeError("schema", f"has invalid {field}")
    return value


def _is_safe_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sequence(value: Any) -> int:
    if not _is_safe_integer(value) or value <= 0:
        raise JsonlDecodeError("schema", "has invalid seq")
    return value


def _require_timestamp(value: Any) -> int:
    if not _is_safe_integer(value) or value < 0:
        raise JsonlDecodeError("schema", "has invalid timestamp")
    return value


def _require_nullable_id(value: Any, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise JsonlDecodeError("schema", f"has invalid {field}")
    return value


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _decode_header(line: str) -> JsonlV4Header:
    value = _parse_object(line)
    if value.get("kind") != "header":
        raise JsonlDecodeError("schema", "is not a header")
    if value.get("version") != 4 or isinstance(value.get("version"), bool):
        raise JsonlDecodeError("schema", "has unsupported session version")
    parent_session_id = value.get("parentSessionId")
    if parent_session_id is not None and not isinstance(parent_session_id, str):
        raise JsonlDecodeError("schema", "has invalid parentSessionId")
    legacy_parent_session_path = value.get("legacyParentSessionPath")
    if legacy_parent_session_path is not None and not isinstance(legacy_parent_session_path, str):
        raise JsonlDecodeError("schema", "has invalid legacyParentSessionPath")
    if parent_session_id is not None and legacy_parent_session_path is not None:
        raise JsonlDecodeError("schema", "has both parentSessionId and legacyParentSessionPath")
    metadata = value.get("metadata")
    if metadata is not None and not _is_object(metadata):
        raise JsonlDecodeError("schema", "has invalid metadata")
    return JsonlV4Header(
        id=_require_string(value.get("id"), "id"),
        created_at=_require_timestamp(value.get("createdAt")),
        cwd=_require_string(value.get("cwd"), "cwd"),
        parent_session_id=parent_session_id,
        legacy_parent_session_path=legacy_parent_session_path,
        metadata=metadata,
    )


def parse_header(line: str) -> Result[JsonlV4Header, JsonlDecodeError]:
    try:
        return ok(_decode_header(line))
    except JsonlDecodeError as error:
        return err(error)


def encode_header(header: JsonlV4Header) -> str:
    data: dict[str, Any] = {
        "kind": "header",
        "version": 4,
        "id": header.id,
        "createdAt": header.created_at,
        "cwd": header.cwd,
    }
    _put(data, "parentSessionId", header.parent_session_id)
    _put(data, "legacyParentSessionPath", header.legacy_parent_session_path)
    _put(data, "metadata", header.metadata)
    return _dump(data)


def metadata_from_header(header: JsonlV4Header, path: str, modified_at: float) -> JsonlSessionMetadata:
    return JsonlSessionMetadata(
        id=header.id,
        created_at=header.created_at,
        cwd=header.cwd,
        path=path,
        modified_at=modified_at,
        source_format=4,
        parent_session_id=header.parent_session_id,
        legacy_parent_session_path=header.legacy_parent_session_path,
        metadata=header.metadata,
    )


def _entry_from_wire(
    value: dict[str, Any],
    entry_type: str,
    *,
    id: str,
    parent_id: str | None,
    seq: int,
    timestamp: int,
) -> Entry:
    if entry_type == "message":
        terminate = value.get("terminate")
        return MessageEntry(
            id=id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            message=parse_message(value.get("message")),
            terminate=True if terminate is True else None,
        )
    if entry_type == "model_change":
        return ModelChangeEntry(
            id=id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            provider=value.get("provider", ""),
            model_id=value.get("modelId", ""),
        )
    if entry_type == "thinking_level_change":
        return ThinkingLevelEntry(
            id=id, parent_id=parent_id, seq=seq, timestamp=timestamp, thinking_level=value.get("thinkingLevel", "")
        )
    if entry_type == "active_tools_change":
        return ActiveToolsEntry(
            id=id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            active_tool_names=value.get("activeToolNames") or [],
        )
    if entry_type == "compaction":
        retained_tail = value.get("retainedTail")
        return CompactionEntry(
            id=id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            summary=value.get("summary", ""),
            retained_tail=(
                [parse_message(message) for message in retained_tail] if isinstance(retained_tail, list) else []
            ),
            tokens_before=value.get("tokensBefore", 0),
            details=value.get("details"),
            usage=parse_usage(value.get("usage")),
        )
    if entry_type == "branch_summary":
        return BranchSummaryEntry(
            id=id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            from_id=value.get("fromId", ""),
            summary=value.get("summary", ""),
            details=value.get("details"),
            usage=parse_usage(value.get("usage")),
        )
    return CustomEntry(
        id=id,
        parent_id=parent_id,
        seq=seq,
        timestamp=timestamp,
        custom_type=value.get("customType", ""),
        data=value.get("data"),
    )


def _entry_to_wire(entry: Entry, *, provisioned: bool) -> dict[str, Any]:
    data: dict[str, Any] = {"type": entry.type, "id": entry.id}
    if entry.type == "message":
        data["message"] = serialize_message(entry.message)
        if entry.terminate is True:
            data["terminate"] = True
    elif entry.type == "model_change":
        data["provider"] = entry.provider
        data["modelId"] = entry.model_id
    elif entry.type == "thinking_level_change":
        data["thinkingLevel"] = entry.thinking_level
    elif entry.type == "active_tools_change":
        data["activeToolNames"] = entry.active_tool_names
    elif entry.type == "compaction":
        data["summary"] = entry.summary
        data["retainedTail"] = [serialize_message(message) for message in entry.retained_tail]
        data["tokensBefore"] = entry.tokens_before
        _put(data, "details", to_wire_value(entry.details))
        _put(data, "usage", serialize_usage(entry.usage))
    elif entry.type == "branch_summary":
        data["fromId"] = entry.from_id
        data["summary"] = entry.summary
        _put(data, "details", to_wire_value(entry.details))
        _put(data, "usage", serialize_usage(entry.usage))
    else:
        data["customType"] = entry.custom_type
        _put(data, "data", entry.data)
    if not provisioned:
        data["parentId"] = entry.parent_id
        data["seq"] = entry.seq
        data["timestamp"] = entry.timestamp
    return data


def _parse_provisioned_entry(value: Any, field: str) -> Entry:
    if not _is_object(value):
        raise JsonlDecodeError("schema", f"has invalid {field}")
    entry_type = _require_string(value.get("type"), f"{field} entry type")
    if entry_type not in ENTRY_TYPES:
        raise JsonlDecodeError("schema", f"has unknown entry type {entry_type}")
    id = _require_string(value.get("id"), f"{field} id")
    if entry_type == "custom":
        _require_string(value.get("customType"), "customType")
    return _entry_from_wire(value, entry_type, id=id, parent_id=None, seq=0, timestamp=0)


def _intent_from_wire(value: dict[str, Any]) -> OperationIntent:
    kind = value.get("kind")
    if kind == "run":
        return RunIntent(
            original_prompt=[parse_message(message) for message in value.get("originalPrompt") or []],
            initial_messages=[
                _parse_provisioned_entry(entry, "initialMessages") for entry in value.get("initialMessages") or []
            ],
            system_prompt_override=value.get("systemPromptOverride"),
            resume_data=value.get("resumeData"),
        )
    if kind == "compaction":
        return CompactionIntent(
            result_entry_id=value.get("resultEntryId", ""), custom_instructions=value.get("customInstructions")
        )
    return NavigationIntent(
        target_id=value.get("targetId"),
        summarize=bool(value.get("summarize", False)),
        custom_instructions=value.get("customInstructions"),
        label=value.get("label"),
        summary_entry_id=value.get("summaryEntryId"),
    )


def _intent_to_wire(intent: OperationIntent) -> dict[str, Any]:
    if intent.kind == "run":
        data: dict[str, Any] = {
            "kind": "run",
            "originalPrompt": [serialize_message(message) for message in intent.original_prompt],
            "initialMessages": [_entry_to_wire(entry, provisioned=True) for entry in intent.initial_messages],
        }
        _put(data, "systemPromptOverride", intent.system_prompt_override)
        _put(data, "resumeData", intent.resume_data)
        return data
    if intent.kind == "compaction":
        data = {"kind": "compaction"}
        _put(data, "customInstructions", intent.custom_instructions)
        data["resultEntryId"] = intent.result_entry_id
        return data
    data = {"kind": "navigation", "targetId": intent.target_id, "summarize": intent.summarize}
    _put(data, "customInstructions", intent.custom_instructions)
    _put(data, "label", intent.label)
    _put(data, "summaryEntryId", intent.summary_entry_id)
    return data


def _record_from_wire(
    value: dict[str, Any],
    record_type: str,
    *,
    id: str,
    lane: str,
    seq: int,
    timestamp: int,
) -> LaneRecord:
    if record_type == "operation_started":
        return OperationStartedRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            source_leaf_id=value.get("sourceLeafId"),
            intent=_intent_from_wire(value["intent"]),
        )
    if record_type == "abort_requested":
        return AbortRequestedRecord(id=id, lane=lane, seq=seq, timestamp=timestamp, run_id=value.get("runId", ""))
    if record_type == "operation_finished":
        return OperationFinishedRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            run_id=value.get("runId", ""),
            outcome=value.get("outcome", "completed"),
            error=value.get("error"),
        )
    if record_type == "step_attempt":
        return StepAttemptRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            run_id=value.get("runId", ""),
            step=value.get("step", "assistant"),
            attempt=value.get("attempt", 0),
            result_entry_id=value.get("resultEntryId", ""),
            compaction_reason=value.get("compactionReason"),
        )
    if record_type == "tool_started":
        return ToolStartedRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            run_id=value.get("runId", ""),
            assistant_entry_id=value.get("assistantEntryId", ""),
            tool_index=value.get("toolIndex", 0),
            tool_call_id=value.get("toolCallId", ""),
            tool_name=value.get("toolName", ""),
            effective_args=value.get("effectiveArgs") or {},
            result_entry_id=value.get("resultEntryId", ""),
            replay=value.get("replay", "never"),
        )
    if record_type == "queue_enqueued":
        return QueueEnqueuedRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            queue=value.get("queue", "nextRun"),
            run_id=value.get("runId"),
            target=_parse_provisioned_entry(value.get("target"), "target"),
        )
    if record_type == "queue_cancelled":
        return QueueCancelledRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            run_id=value.get("runId"),
            entry_id=value.get("entryId", ""),
        )
    if record_type == "write_deferred":
        return WriteDeferredRecord(
            id=id,
            lane=lane,
            seq=seq,
            timestamp=timestamp,
            run_id=value.get("runId", ""),
            target=_parse_provisioned_entry(value.get("target"), "target"),
        )
    return UsageRecord(
        id=id,
        lane=lane,
        seq=seq,
        timestamp=timestamp,
        cause=value.get("cause", "adjustment"),
        usage=parse_usage(value.get("usage")),
        run_id=value.get("runId"),
        entry_id=value.get("entryId"),
        attempt=value.get("attempt"),
        stop_reason=value.get("stopReason"),
        tool_call_id=value.get("toolCallId"),
        details=value.get("details"),
    )


def _record_to_wire(record: LaneRecord) -> dict[str, Any]:
    data: dict[str, Any] = {"type": record.type, "id": record.id, "lane": record.lane}
    if record.type == "operation_started":
        data["sourceLeafId"] = record.source_leaf_id
        data["intent"] = _intent_to_wire(record.intent)
    elif record.type == "abort_requested":
        data["runId"] = record.run_id
    elif record.type == "operation_finished":
        data["runId"] = record.run_id
        data["outcome"] = record.outcome
        _put(data, "error", record.error)
    elif record.type == "step_attempt":
        data["runId"] = record.run_id
        data["step"] = record.step
        data["attempt"] = record.attempt
        data["resultEntryId"] = record.result_entry_id
        _put(data, "compactionReason", record.compaction_reason)
    elif record.type == "tool_started":
        data["runId"] = record.run_id
        data["assistantEntryId"] = record.assistant_entry_id
        data["toolIndex"] = record.tool_index
        data["toolCallId"] = record.tool_call_id
        data["toolName"] = record.tool_name
        data["effectiveArgs"] = record.effective_args
        data["resultEntryId"] = record.result_entry_id
        data["replay"] = record.replay
    elif record.type == "queue_enqueued":
        data["queue"] = record.queue
        _put(data, "runId", record.run_id)
        data["target"] = _entry_to_wire(record.target, provisioned=True)
    elif record.type == "queue_cancelled":
        _put(data, "runId", record.run_id)
        data["entryId"] = record.entry_id
    elif record.type == "write_deferred":
        data["runId"] = record.run_id
        data["target"] = _entry_to_wire(record.target, provisioned=True)
    else:
        data["cause"] = record.cause
        data["usage"] = serialize_usage(record.usage)
        _put(data, "runId", record.run_id)
        _put(data, "entryId", record.entry_id)
        _put(data, "attempt", record.attempt)
        _put(data, "stopReason", record.stop_reason)
        _put(data, "toolCallId", record.tool_call_id)
        _put(data, "details", record.details)
    data["seq"] = record.seq
    data["timestamp"] = record.timestamp
    return data


def _parse_entry_mutation(value: dict[str, Any], seq: int) -> EntryMutation:
    lane = value.get("lane")
    if lane is not None:
        lane = _require_string(lane, "lane")
    id = _require_string(value.get("id"), "id")
    entry_type = _require_string(value.get("type"), "entry type")
    if entry_type not in ENTRY_TYPES:
        raise JsonlDecodeError("schema", f"has unknown entry type {entry_type}")
    parent_id = _require_nullable_id(value.get("parentId"), "parentId")
    timestamp = _require_timestamp(value.get("timestamp"))
    if entry_type == "custom":
        _require_string(value.get("customType"), "customType")
    entry = _entry_from_wire(value, entry_type, id=id, parent_id=parent_id, seq=seq, timestamp=timestamp)
    return EntryMutation(entry=entry) if lane is None else EntryMutation(lane=lane, entry=entry)


def _parse_record_mutation(value: dict[str, Any], seq: int) -> RecordMutation:
    id = _require_string(value.get("id"), "id")
    lane = _require_string(value.get("lane"), "lane")
    record_type = _require_string(value.get("type"), "record type")
    if record_type not in RECORD_TYPES:
        raise JsonlDecodeError("schema", f"has unknown record type {record_type}")
    timestamp = _require_timestamp(value.get("timestamp"))
    if record_type == "operation_started":
        if not _is_object(value.get("intent")):
            raise JsonlDecodeError("schema", "has invalid intent")
        operation_kind = _require_string(value["intent"].get("kind"), "operation kind")
        if operation_kind not in OPERATION_KINDS:
            raise JsonlDecodeError("schema", f"has unknown operation kind {operation_kind}")
    if record_type == "operation_finished":
        _require_string(value.get("runId"), "runId")
    return RecordMutation(record=_record_from_wire(value, record_type, id=id, lane=lane, seq=seq, timestamp=timestamp))


def _parse_lane_mutation(value: dict[str, Any], seq: int) -> LaneMutation:
    return LaneMutation(
        seq=seq,
        lane=_require_string(value.get("lane"), "lane"),
        leaf_id=_require_nullable_id(value.get("leafId"), "leafId"),
    )


def _parse_fact_mutation(value: dict[str, Any], seq: int) -> NameFactMutation | LabelFactMutation:
    fact = value.get("fact")
    if fact == "name":
        name = value.get("name")
        if name is not None and not isinstance(name, str):
            raise JsonlDecodeError("schema", "has invalid name")
        return NameFactMutation(seq=seq, name=name)
    if fact == "label":
        label = value.get("label")
        if label is not None and not isinstance(label, str):
            raise JsonlDecodeError("schema", "has invalid label")
        return LabelFactMutation(
            seq=seq,
            target_id=_require_string(value.get("targetId"), "targetId"),
            label=label,
        )
    raise JsonlDecodeError("schema", "has unknown fact type")


def _decode_mutation(line: str) -> SessionMutation:
    value = _parse_object(line)
    seq = _require_sequence(value.get("seq"))
    kind = value.get("kind")
    if kind == "entry":
        return _parse_entry_mutation(value, seq)
    if kind == "record":
        return _parse_record_mutation(value, seq)
    if kind == "lane":
        return _parse_lane_mutation(value, seq)
    if kind == "fact":
        return _parse_fact_mutation(value, seq)
    raise JsonlDecodeError("schema", "has unknown mutation kind")


def parse_mutation(line: str) -> Result[SessionMutation, JsonlDecodeError]:
    try:
        return ok(_decode_mutation(line))
    except JsonlDecodeError as error:
        return err(error)


def encode_mutation(mutation: SessionMutation) -> str:
    if mutation.kind == "entry":
        data: dict[str, Any] = {"kind": "entry"}
        _put(data, "lane", mutation.lane)
        data.update(_entry_to_wire(mutation.entry, provisioned=False))
        return _dump(data)
    if mutation.kind == "record":
        return _dump({"kind": "record", **_record_to_wire(mutation.record)})
    if mutation.kind == "lane":
        return _dump({"kind": "lane", "seq": mutation.seq, "lane": mutation.lane, "leafId": mutation.leaf_id})
    if mutation.fact == "name":
        data = {"kind": "fact", "seq": mutation.seq, "fact": "name"}
        # A cleared name is dropped from the wire like JSON.stringify drops undefined.
        _put(data, "name", mutation.name)
        return _dump(data)
    data = {"kind": "fact", "seq": mutation.seq, "fact": "label", "targetId": mutation.target_id}
    _put(data, "label", mutation.label)
    return _dump(data)
