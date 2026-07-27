"""Mirror of pi coding-agent test/tree-selector.test.ts."""

import time
from datetime import UTC, datetime

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import SessionTreeNode
from pidrei.modes.interactive.components import TreeSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import AssistantMessage, TextContent, ToolCall, Usage, UserMessage
from pidrei_tui import set_keybindings, visible_width


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Ensure test isolation: keybindings are a global singleton
    set_keybindings(KeybindingsManager())


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def user_message(entry_id: str, parent_id: str | None, content: str) -> dict:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": _now_iso(),
        "message": UserMessage(content=content, timestamp=int(time.time() * 1000)),
    }


def assistant_message(entry_id: str, parent_id: str | None, text: str) -> dict:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": _now_iso(),
        "message": AssistantMessage(
            content=[TextContent(text=text)],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-sonnet-4",
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        ),
    }


def tool_call_only_assistant(entry_id: str, parent_id: str | None) -> dict:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": _now_iso(),
        "message": AssistantMessage(
            content=[ToolCall(id=f"tc-{entry_id}", name="read", arguments={"path": "test.ts"})],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-sonnet-4",
            usage=Usage(),
            stop_reason="toolUse",
            timestamp=int(time.time() * 1000),
        ),
    }


def model_change(entry_id: str, parent_id: str | None) -> dict:
    return {
        "type": "model_change",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": _now_iso(),
        "provider": "anthropic",
        "modelId": "claude-sonnet-4",
    }


def build_tree(entries: list) -> list:
    if not entries:
        return []

    nodes = [SessionTreeNode(entry=entry, children=[]) for entry in entries]

    by_id = {node.entry["id"]: node for node in nodes}

    roots: list = []
    for node in nodes:
        if node.entry["parentId"] is None:
            roots.append(node)
        else:
            parent = by_id.get(node.entry["parentId"])
            if parent is not None:
                parent.children.append(node)
    return roots


def _make_selector(tree, current_leaf_id, on_label_change=None):
    return TreeSelectorComponent(
        tree,
        current_leaf_id,
        24,
        lambda entry_id: None,
        lambda: None,
        on_label_change,
    )


class TestInitialSelectionWithMetadataEntries:
    def test_focuses_nearest_visible_ancestor_when_current_leaf_id_is_a_model_change_with_sibling_branch(self):
        # Tree structure:
        # user-1
        # └── asst-1
        #     ├── user-2 (active branch)
        #     │   └── model-1 (model_change, CURRENT LEAF)
        #     └── user-3 (sibling branch, added later chronologically)
        entries = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),  # Active branch
            model_change("model-1", "user-2"),  # Current leaf (metadata)
            user_message("user-3", "asst-1", "sibling branch"),  # Sibling branch
        ]
        tree = build_tree(entries)

        selector = _make_selector(tree, "model-1")

        tree_list = selector.get_tree_list()
        # Should focus on user-2 (parent of model-1), not user-3 (last item)
        assert tree_list.get_selected_node().entry["id"] == "user-2"

    def test_focuses_nearest_visible_ancestor_when_current_leaf_id_is_a_thinking_level_change_entry(self):
        # Similar structure with thinking_level_change instead of model_change
        entries = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),
            {
                "type": "thinking_level_change",
                "id": "thinking-1",
                "parentId": "user-2",
                "timestamp": _now_iso(),
                "thinkingLevel": "high",
            },
            user_message("user-3", "asst-1", "sibling branch"),
        ]
        tree = build_tree(entries)

        selector = _make_selector(tree, "thinking-1")

        tree_list = selector.get_tree_list()
        assert tree_list.get_selected_node().entry["id"] == "user-2"


class TestFilterSwitchingWithParentTraversal:
    @pytest.mark.tonio
    async def test_switches_to_nearest_visible_user_message_when_changing_to_user_only_filter(self):
        # In user-only filter: [user-1, user-2, user-3]
        entries = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),
            assistant_message("asst-2", "user-2", "response"),
            user_message("user-3", "asst-1", "sibling branch"),
        ]
        tree = build_tree(entries)

        selector = _make_selector(tree, "asst-2")

        tree_list = selector.get_tree_list()
        assert tree_list.get_selected_node().entry["id"] == "asst-2"

        # Simulate Ctrl+U (user-only filter)
        await selector.handle_input("\x15")

        # Should now be on user-2 (the parent user message), not user-3
        assert tree_list.get_selected_node().entry["id"] == "user-2"

    @pytest.mark.tonio
    async def test_returns_to_nearest_visible_ancestor_when_switching_back_to_default_filter(self):
        # Same branching structure
        entries = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),
            assistant_message("asst-2", "user-2", "response"),
            user_message("user-3", "asst-1", "sibling branch"),
        ]
        tree = build_tree(entries)

        selector = _make_selector(tree, "asst-2")

        tree_list = selector.get_tree_list()
        assert tree_list.get_selected_node().entry["id"] == "asst-2"

        # Switch to user-only
        await selector.handle_input("\x15")  # Ctrl+U
        assert tree_list.get_selected_node().entry["id"] == "user-2"

        # Switch back to default - should stay on user-2
        # (since that's what we navigated to via parent traversal)
        await selector.handle_input("\x04")  # Ctrl+D
        assert tree_list.get_selected_node().entry["id"] == "user-2"


class TestHelp:
    def test_renders_semantic_help_rows_without_truncating_narrow_terminal_controls(self):
        entries = [user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", "hi")]
        tree = build_tree(entries)
        selector = _make_selector(tree, "asst-1")

        plain_lines = [strip_ansi(line) for line in selector.render(30)]
        plain = "\n".join(plain_lines)
        assert "branch" in plain
        assert "copy" in plain
        assert "filters" in plain
        assert "cycle" in plain
        assert "label time" in plain
        assert "..." not in plain
        assert all(visible_width(line) <= 30 for line in plain_lines)


class TestCopy:
    @pytest.mark.tonio
    async def test_copies_the_full_selected_message_with_ctrl_x(self):
        message = "long message " * 30 + "\nsecond line"
        tree = build_tree([user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", message)])
        selector = _make_selector(tree, "asst-1")
        copied = None

        def on_copy(text):
            nonlocal copied
            copied = text

        selector.on_copy = on_copy

        await selector.handle_input("\x18")

        assert copied == message


class TestLabelTimestamps:
    @pytest.mark.tonio
    async def test_toggles_label_timestamps_for_labeled_nodes(self):
        entries = [user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", "hi")]
        tree = build_tree(entries)
        label_date = datetime(2026, 3, 28, 14, 32, 0).astimezone()
        tree[0].label = "checkpoint"
        tree[0].label_timestamp = label_date.isoformat()

        selector = _make_selector(tree, "asst-1")

        tree_list = selector.get_tree_list()
        render = "\n".join(tree_list.render(200))
        assert "[checkpoint]" in render
        assert "3/28 14:32" not in render
        assert "[+label time]" not in render

        await selector.handle_input("T")

        render = "\n".join(tree_list.render(200))
        assert "3/28 14:32" in render
        assert "[+label time]" in render


class TestEmptyFilterPreservation:
    @pytest.mark.tonio
    async def test_preserves_selection_when_switching_to_empty_labeled_filter_and_back(self):
        # Tree with no labels
        entries = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "bye"),
            assistant_message("asst-2", "user-2", "goodbye"),
        ]
        tree = build_tree(entries)

        selector = _make_selector(tree, "asst-2")

        tree_list = selector.get_tree_list()
        assert tree_list.get_selected_node().entry["id"] == "asst-2"

        # Switch to labeled-only filter (no labels exist, so empty result)
        await selector.handle_input("\x0c")  # Ctrl+L

        # The list should be empty, get_selected_node returns None
        assert tree_list.get_selected_node() is None

        # Switch back to default filter
        await selector.handle_input("\x04")  # Ctrl+D

        # Should restore to asst-2 (the selection before the empty filter)
        assert tree_list.get_selected_node().entry["id"] == "asst-2"

    @pytest.mark.tonio
    async def test_preserves_selection_through_multiple_empty_filter_switches(self):
        entries = [user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", "hi")]
        tree = build_tree(entries)

        selector = _make_selector(tree, "asst-1")

        tree_list = selector.get_tree_list()
        assert tree_list.get_selected_node().entry["id"] == "asst-1"

        # Switch to labeled-only (empty) - Ctrl+L toggles labeled ↔ default
        await selector.handle_input("\x0c")  # Ctrl+L -> labeled-only
        assert tree_list.get_selected_node() is None

        # Switch to default, then back to labeled-only
        await selector.handle_input("\x0c")  # Ctrl+L -> default (toggle back)
        assert tree_list.get_selected_node().entry["id"] == "asst-1"

        await selector.handle_input("\x0c")  # Ctrl+L -> labeled-only again
        assert tree_list.get_selected_node() is None

        # Switch back to default with Ctrl+D
        await selector.handle_input("\x04")  # Ctrl+D
        assert tree_list.get_selected_node().entry["id"] == "asst-1"


UP = "\x1b[A"
DOWN = "\x1b[B"
CTRL_LEFT = "\x1b[1;5D"
CTRL_RIGHT = "\x1b[1;5C"
ALT_LEFT = "\x1b[1;3D"
ALT_RIGHT = "\x1b[1;3C"


def build_branching_tree() -> list:
    # Tree structure:
    #
    # user-1
    # asst-1
    # user-2
    # asst-2          ← branch point (has 2 children)
    # ├─ user-3a      ← branch A (active: leaf is asst-4a)
    # │  asst-3a
    # │  user-4a
    # │  asst-4a
    # └─ user-3b      ← branch B
    #    asst-3b
    #    user-4b
    #
    # Foldable: user-1 (root), user-3a (segment start), user-3b (segment start)
    entries = [
        user_message("user-1", None, "first message"),
        assistant_message("asst-1", "user-1", "response 1"),
        user_message("user-2", "asst-1", "second message"),
        assistant_message("asst-2", "user-2", "response 2"),
        # Branch A (active)
        user_message("user-3a", "asst-2", "branch A start"),
        assistant_message("asst-3a", "user-3a", "branch A response"),
        user_message("user-4a", "asst-3a", "branch A deep"),
        assistant_message("asst-4a", "user-4a", "branch A leaf"),
        # Branch B
        user_message("user-3b", "asst-2", "branch B start"),
        assistant_message("asst-3b", "user-3b", "branch B response"),
        user_message("user-4b", "asst-3b", "branch B deep"),
    ]
    return build_tree(entries)


class TestBranchNavigationAndFoldingWithCtrlArrowKeys:
    @pytest.mark.tonio
    async def test_ctrl_right_unfolds_a_folded_node_then_does_segment_jump_when_unfolded(self):
        tree = build_branching_tree()
        selector = _make_selector(tree, "asst-4a")
        tree_list = selector.get_tree_list()

        await selector.handle_input(CTRL_LEFT)  # asst-4a → user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(CTRL_LEFT)  # fold user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(DOWN)  # user-3a → user-3b (children hidden)
        assert tree_list.get_selected_node().entry["id"] == "user-3b"

        await selector.handle_input(UP)  # user-3b → user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(CTRL_RIGHT)  # unfold user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(DOWN)  # user-3a → asst-3a (children restored)
        assert tree_list.get_selected_node().entry["id"] == "asst-3a"

        await selector.handle_input(CTRL_LEFT)  # asst-3a → user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(CTRL_RIGHT)  # user-3a → asst-4a (segment jump to leaf)
        assert tree_list.get_selected_node().entry["id"] == "asst-4a"

    @pytest.mark.tonio
    async def test_alt_left_right_are_aliases_for_fold_and_unfold_navigation(self):
        tree = build_branching_tree()
        selector = _make_selector(tree, "asst-4a")
        tree_list = selector.get_tree_list()

        await selector.handle_input(ALT_LEFT)  # asst-4a → user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(ALT_LEFT)  # fold user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(ALT_RIGHT)  # unfold user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(ALT_RIGHT)  # user-3a → asst-4a
        assert tree_list.get_selected_node().entry["id"] == "asst-4a"

    @pytest.mark.tonio
    async def test_folding_root_hides_entire_subtree_nested_fold_preserved_on_unfold(self):
        tree = build_branching_tree()
        selector = _make_selector(tree, "asst-4a")
        tree_list = selector.get_tree_list()

        await selector.handle_input(CTRL_LEFT)  # asst-4a → user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(CTRL_LEFT)  # fold user-3a
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(CTRL_LEFT)  # user-3a (folded) → user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(CTRL_LEFT)  # fold user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(DOWN)  # wrap (only visible node)
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(CTRL_RIGHT)  # unfold user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(CTRL_RIGHT)  # user-1 → user-3a (segment jump, user-3a still folded)
        assert tree_list.get_selected_node().entry["id"] == "user-3a"

        await selector.handle_input(DOWN)  # user-3a → user-3b (user-3a still folded)
        assert tree_list.get_selected_node().entry["id"] == "user-3b"

    @pytest.mark.tonio
    async def test_fold_and_navigate_on_non_active_branch(self):
        tree = build_branching_tree()
        selector = _make_selector(tree, "asst-4a")
        tree_list = selector.get_tree_list()

        # Navigate down to user-3b (branch B)
        found = False
        for _ in range(20):
            await selector.handle_input(DOWN)
            if tree_list.get_selected_node().entry["id"] == "user-3b":
                found = True
                break
        assert found is True

        await selector.handle_input(CTRL_RIGHT)  # user-3b → user-4b (segment jump to leaf)
        assert tree_list.get_selected_node().entry["id"] == "user-4b"

        await selector.handle_input(CTRL_LEFT)  # user-4b → user-3b
        assert tree_list.get_selected_node().entry["id"] == "user-3b"

        await selector.handle_input(CTRL_LEFT)  # fold user-3b
        assert tree_list.get_selected_node().entry["id"] == "user-3b"

        await selector.handle_input(CTRL_LEFT)  # user-3b (folded) → user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

    @pytest.mark.tonio
    async def test_fold_and_navigate_with_multiple_roots(self):
        entries = [
            user_message("user-1", None, "first root"),
            assistant_message("asst-1", "user-1", "response 1"),
            user_message("user-2", None, "second root"),
            assistant_message("asst-2", "user-2", "response 2"),
        ]
        tree = build_tree(entries)
        selector = _make_selector(tree, "asst-1")
        tree_list = selector.get_tree_list()

        assert tree_list.get_selected_node().entry["id"] == "asst-1"

        await selector.handle_input(CTRL_LEFT)  # asst-1 → user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(CTRL_LEFT)  # fold user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(DOWN)  # user-1 → user-2 (children hidden)
        assert tree_list.get_selected_node().entry["id"] == "user-2"

        await selector.handle_input(CTRL_RIGHT)  # user-2 → asst-2 (segment jump to leaf)
        assert tree_list.get_selected_node().entry["id"] == "asst-2"

        await selector.handle_input(CTRL_LEFT)  # asst-2 → user-2
        assert tree_list.get_selected_node().entry["id"] == "user-2"

        await selector.handle_input(CTRL_LEFT)  # fold user-2
        assert tree_list.get_selected_node().entry["id"] == "user-2"

        await selector.handle_input(CTRL_LEFT)  # user-2 (folded, root) → stays on user-2
        assert tree_list.get_selected_node().entry["id"] == "user-2"

    @pytest.mark.tonio
    async def test_folding_root_hides_descendants_even_when_intermediate_nodes_are_filtered_out(self):
        # user-1 → toolCallOnly-1 (filtered out) → user-2 → asst-2
        entries = [
            user_message("user-1", None, "hello"),
            tool_call_only_assistant("tool-asst-1", "user-1"),
            user_message("user-2", "tool-asst-1", "follow up"),
            assistant_message("asst-2", "user-2", "response"),
        ]
        tree = build_tree(entries)
        selector = _make_selector(tree, "asst-2")
        tree_list = selector.get_tree_list()

        await selector.handle_input(CTRL_LEFT)  # asst-2 → user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(CTRL_LEFT)  # fold user-1
        assert tree_list.get_selected_node().entry["id"] == "user-1"

        await selector.handle_input(DOWN)  # wrap (only visible node)
        assert tree_list.get_selected_node().entry["id"] == "user-1"

    @pytest.mark.tonio
    async def test_search_resets_fold_state(self):
        tree = build_branching_tree()
        selector = _make_selector(tree, "asst-4a")
        tree_list = selector.get_tree_list()

        await selector.handle_input(CTRL_LEFT)  # asst-4a → user-3a
        await selector.handle_input(CTRL_LEFT)  # fold user-3a

        await selector.handle_input(DOWN)  # user-3a → user-3b (children hidden)
        assert tree_list.get_selected_node().entry["id"] == "user-3b"

        await selector.handle_input("b")  # search resets folds
        await selector.handle_input("\x1b")  # clear search

        # Navigate to user-3a to verify fold was reset
        current_id = ""
        for _ in range(20):
            await selector.handle_input(DOWN)
            node = tree_list.get_selected_node()
            current_id = node.entry["id"] if node is not None else ""
            if current_id == "user-3a":
                break
        assert current_id == "user-3a"

        await selector.handle_input(DOWN)  # user-3a → asst-3a (not user-3b)
        assert tree_list.get_selected_node().entry["id"] == "asst-3a"

    @pytest.mark.tonio
    async def test_filter_mode_change_resets_fold_state(self):
        tree = build_branching_tree()
        selector = _make_selector(tree, "asst-4a")
        tree_list = selector.get_tree_list()

        await selector.handle_input(CTRL_LEFT)  # asst-4a → user-3a
        await selector.handle_input(CTRL_LEFT)  # fold user-3a

        await selector.handle_input("\x15")  # ctrl+u: user-only filter resets folds
        await selector.handle_input("\x04")  # ctrl+d: back to default

        # Navigate to user-3a to verify fold was reset
        current_id = ""
        for _ in range(20):
            await selector.handle_input(DOWN)
            node = tree_list.get_selected_node()
            current_id = node.entry["id"] if node is not None else ""
            if current_id == "user-3a":
                break
        assert current_id == "user-3a"

        await selector.handle_input(DOWN)  # user-3a → asst-3a (not user-3b)
        assert tree_list.get_selected_node().entry["id"] == "asst-3a"
