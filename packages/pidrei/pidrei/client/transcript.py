"""Remote transcript projection (port of pi coding-agent `client/transcript.ts`).

A pure reducer over protocol snapshots and progress events: the authoritative
`SessionSnapshot` is never mutated, streamed deltas project on top of it, and
`select_transcript` merges the two views. Protocol values are the camelCase
dicts produced by `pidrei_protocol`; JS `structuredClone` maps to
`copy.deepcopy`.

`parse_partial_tool_input` mirrors `JSON.parse` + the `isJsonValue` guard:
Python's `json.loads` accepts `NaN`/`Infinity` where `JSON.parse` throws, but
the guard rejects non-finite numbers either way, so both implementations
preserve the raw prefix for exactly the same inputs.
"""

import copy
import json
import math
from dataclasses import dataclass, replace
from typing import Any

from pidrei_protocol import SessionSnapshot, TranscriptItem, TranscriptProgress


@dataclass(slots=True, frozen=True)
class TranscriptState:
    snapshot: SessionSnapshot
    progress_items: dict[str, TranscriptItem]
    progress_order: tuple[str, ...]
    tool_call_buffers: dict[str, str]


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if type(value) is not dict:
        return False
    return all(_is_json_value(item) for item in value.values())


def parse_partial_tool_input(value: str) -> Any:
    try:
        parsed = json.loads(value)
        if _is_json_value(parsed):
            return parsed
    except ValueError:
        # Tool arguments are incomplete while streaming. Preserve their raw prefix until they form valid JSON.
        pass
    return value


def create_transcript_state(snapshot: SessionSnapshot) -> TranscriptState:
    return TranscriptState(
        snapshot=copy.deepcopy(snapshot),
        progress_items={},
        progress_order=(),
        tool_call_buffers={},
    )


def apply_transcript_snapshot(state: TranscriptState, snapshot: SessionSnapshot) -> TranscriptState:
    if state.snapshot["id"] == snapshot["id"] and snapshot["revision"] < state.snapshot["revision"]:
        return state
    return create_transcript_state(snapshot)


def apply_transcript_progress(state: TranscriptState, progress: TranscriptProgress) -> TranscriptState:
    if progress["type"] in ("item_started", "item_updated"):
        return _set_progress_item(state, progress["item"])
    if progress["type"] == "item_finished":
        tool_call_buffers = dict(state.tool_call_buffers)
        prefix = f"{progress['item']['id']}:"
        for key in list(tool_call_buffers):
            if key.startswith(prefix):
                del tool_call_buffers[key]
        return _set_progress_item(replace(state, tool_call_buffers=tool_call_buffers), progress["item"])

    item = state.progress_items.get(progress["messageId"])
    if item is None:
        item = next(
            (candidate for candidate in state.snapshot["transcript"] if candidate["id"] == progress["messageId"]),
            None,
        )
    if item is None or item["role"] != "assistant":
        return state
    tool_call_buffers = state.tool_call_buffers
    content: list[Any] = []
    for index, part in enumerate(item["content"]):
        if index != progress["contentIndex"]:
            content.append(copy.deepcopy(part))
        elif progress["kind"] == "text" and part["type"] == "text":
            content.append({**part, "text": part["text"] + progress["delta"]})
        elif progress["kind"] == "thinking" and part["type"] == "thinking":
            content.append({**part, "thinking": part["thinking"] + progress["delta"]})
        elif progress["kind"] == "toolCall" and part["type"] == "toolCall":
            key = f"{progress['messageId']}:{progress['contentIndex']}"
            existing = state.tool_call_buffers.get(key)
            if existing is None:
                existing = part["input"] if isinstance(part["input"], str) else ""
            buffer = existing + progress["delta"]
            tool_call_buffers = {**state.tool_call_buffers, key: buffer}
            content.append({**part, "input": parse_partial_tool_input(buffer)})
        else:
            content.append(copy.deepcopy(part))
    return _set_progress_item(replace(state, tool_call_buffers=tool_call_buffers), {**item, "content": content})


def select_transcript(state: TranscriptState) -> list[TranscriptItem]:
    transcript = [state.progress_items.get(item["id"], item) for item in state.snapshot["transcript"]]
    ids = {item["id"] for item in transcript}
    for item_id in state.progress_order:
        if item_id in ids:
            continue
        item = state.progress_items.get(item_id)
        if item is not None:
            transcript.append(item)
            ids.add(item_id)
    for item in state.snapshot["queuedSteer"]:
        if item["id"] in ids:
            continue
        transcript.append(item)
        ids.add(item["id"])
    return transcript


def _set_progress_item(state: TranscriptState, item: TranscriptItem) -> TranscriptState:
    progress_items = dict(state.progress_items)
    progress_order = state.progress_order if item["id"] in progress_items else (*state.progress_order, item["id"])
    progress_items[item["id"]] = copy.deepcopy(item)
    return replace(state, progress_items=progress_items, progress_order=progress_order)
