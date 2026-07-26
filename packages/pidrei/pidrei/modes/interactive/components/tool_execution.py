"""Mirror of pi coding-agent src/modes/interactive/components/tool-execution.ts."""

import json

import tonio.colored as tonio

from pidrei_tui import Box, Container, Image, Spacer, Text, get_capabilities

from ....utils.image_process import convert_to_png
from ..theme import theme


def _block_type(block) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _block_get(block, key: str):
    if isinstance(block, dict):
        return block.get(key)
    snake = {"mimeType": "mime_type"}.get(key, key)
    return getattr(block, snake, None)


class ToolExecutionComponent(Container):
    """Options: ``{"showImages"?, "imageWidthCells"?}``."""

    def __init__(self, tool_name: str, tool_call_id: str, args, options, tool_definition, ui, cwd: str) -> None:
        super().__init__()
        options = options or {}
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._args = args
        self._tool_definition = tool_definition
        # Imported lazily: core.tools imports interactive components for its
        # renderers, so a module-level import here would be circular.
        from ....core.tools import create_all_tool_definitions

        self._built_in_tool_definition = create_all_tool_definitions(cwd).get(tool_name)
        show_images = options.get("showImages")
        self._show_images = show_images if show_images is not None else True
        image_width_cells = options.get("imageWidthCells")
        self._image_width_cells = image_width_cells if image_width_cells is not None else 60
        self._ui = ui
        self._cwd = cwd

        self._call_renderer_component = None
        self._result_renderer_component = None
        self._renderer_state: dict = {}
        self._image_components: list = []
        self._image_spacers: list = []
        self._expanded = False
        self._is_partial = True
        self._execution_started = False
        self._args_complete = False
        self._result = None
        self._converted_images: dict = {}
        self._hide_component = False

        self.add_child(Spacer(1))

        # Always create all shell variants. content_box is used for default
        # renderer-based composition. self_render_container is used when the
        # tool renders its own framing. content_text is reserved for generic
        # fallback rendering when no tool definition exists.
        self._content_box = Box(1, 1, lambda text: theme.bg("toolPendingBg", text))
        self._content_text = Text("", 1, 1, lambda text: theme.bg("toolPendingBg", text))
        self._self_render_container = Container()

        if self._has_renderer_definition():
            self.add_child(self._self_render_container if self._get_render_shell() == "self" else self._content_box)
        else:
            self.add_child(self._content_text)

        self._update_display()

    def _get_call_renderer(self):
        if self._built_in_tool_definition is None:
            return getattr(self._tool_definition, "render_call", None)
        if self._tool_definition is None:
            return self._built_in_tool_definition.render_call
        return getattr(self._tool_definition, "render_call", None) or self._built_in_tool_definition.render_call

    def _get_result_renderer(self):
        if self._built_in_tool_definition is None:
            return getattr(self._tool_definition, "render_result", None)
        if self._tool_definition is None:
            return self._built_in_tool_definition.render_result
        return getattr(self._tool_definition, "render_result", None) or self._built_in_tool_definition.render_result

    def _has_renderer_definition(self) -> bool:
        return self._built_in_tool_definition is not None or self._tool_definition is not None

    def _get_render_shell(self) -> str:
        if self._built_in_tool_definition is None:
            return getattr(self._tool_definition, "render_shell", None) or "default"
        if self._tool_definition is None:
            return self._built_in_tool_definition.render_shell or "default"
        return (
            getattr(self._tool_definition, "render_shell", None)
            or self._built_in_tool_definition.render_shell
            or "default"
        )

    def _get_render_context(self, last_component) -> dict:
        def invalidate() -> None:
            self.invalidate()
            self._ui.request_render()

        return {
            "args": self._args,
            "toolCallId": self._tool_call_id,
            "invalidate": invalidate,
            "lastComponent": last_component,
            "state": self._renderer_state,
            "cwd": self._cwd,
            "executionStarted": self._execution_started,
            "argsComplete": self._args_complete,
            "isPartial": self._is_partial,
            "expanded": self._expanded,
            "showImages": self._show_images,
            "isError": bool(self._result["isError"]) if self._result else False,
        }

    def _create_call_fallback(self):
        return Text(theme.fg("toolTitle", theme.bold(self._tool_name)), 0, 0)

    def _create_result_fallback(self):
        output = self._get_text_output()
        if not output:
            return None
        return Text(theme.fg("toolOutput", output), 0, 0)

    def update_args(self, args) -> None:
        self._args = args
        self._update_display()

    def mark_execution_started(self) -> None:
        self._execution_started = True
        self._update_display()
        self._ui.request_render()

    def set_args_complete(self) -> None:
        self._args_complete = True
        self._update_display()
        self._ui.request_render()

    def update_result(self, result: dict, is_partial: bool = False) -> None:
        """``result`` is ``{"content": [...], "details"?, "isError"}``."""
        self._result = result
        self._is_partial = is_partial
        self._update_display()
        self._maybe_convert_images_for_kitty()

    def _maybe_convert_images_for_kitty(self) -> None:
        caps = get_capabilities()
        if caps["images"] != "kitty":
            return
        if not self._result:
            return

        image_blocks = [c for c in self._result["content"] if _block_type(c) == "image"]
        for i, img in enumerate(image_blocks):
            if not _block_get(img, "data") or not _block_get(img, "mimeType"):
                continue
            if _block_get(img, "mimeType") == "image/png":
                continue
            if i in self._converted_images:
                continue

            async def convert(index=i, block=img) -> None:
                converted = await tonio.spawn_blocking(
                    convert_to_png, _block_get(block, "data"), _block_get(block, "mimeType")
                )
                if converted:
                    self._converted_images[index] = converted
                    self._update_display()
                    self._ui.request_render()

            tonio.spawn.without_tracking(convert())

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._update_display()

    def set_show_images(self, show: bool) -> None:
        self._show_images = show
        self._update_display()

    def set_image_width_cells(self, width: float) -> None:
        self._image_width_cells = max(1, int(width))
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def render(self, width: int) -> list:
        if self._hide_component:
            return []

        if self._has_renderer_definition() and self._get_render_shell() == "self":
            content_lines = self._self_render_container.render(width)
            if not content_lines and not self._image_components:
                return []

            lines: list = []
            if content_lines:
                lines.append("")
                lines.extend(content_lines)
            for spacer, image_component in zip(self._image_spacers, self._image_components, strict=False):
                lines.extend(spacer.render(width))
                lines.extend(image_component.render(width))
            return lines

        return super().render(width)

    def _update_display(self) -> None:
        if self._is_partial:
            bg_fn = lambda text: theme.bg("toolPendingBg", text)
        elif self._result and self._result["isError"]:
            bg_fn = lambda text: theme.bg("toolErrorBg", text)
        else:
            bg_fn = lambda text: theme.bg("toolSuccessBg", text)

        has_content = False
        self._hide_component = False
        if self._has_renderer_definition():
            render_container = self._self_render_container if self._get_render_shell() == "self" else self._content_box
            if isinstance(render_container, Box):
                render_container.set_bg_fn(bg_fn)
            render_container.clear()

            call_renderer = self._get_call_renderer()
            if call_renderer is None:
                render_container.add_child(self._create_call_fallback())
                has_content = True
            else:
                try:
                    component = call_renderer(
                        self._args, theme, self._get_render_context(self._call_renderer_component)
                    )
                    self._call_renderer_component = component
                    render_container.add_child(component)
                    has_content = True
                except Exception:
                    self._call_renderer_component = None
                    render_container.add_child(self._create_call_fallback())
                    has_content = True

            if self._result:
                result_renderer = self._get_result_renderer()
                if result_renderer is None:
                    component = self._create_result_fallback()
                    if component is not None:
                        render_container.add_child(component)
                        has_content = True
                else:
                    try:
                        component = result_renderer(
                            {"content": self._result["content"], "details": self._result.get("details")},
                            {"expanded": self._expanded, "isPartial": self._is_partial},
                            theme,
                            self._get_render_context(self._result_renderer_component),
                        )
                        self._result_renderer_component = component
                        render_container.add_child(component)
                        has_content = True
                    except Exception:
                        self._result_renderer_component = None
                        component = self._create_result_fallback()
                        if component is not None:
                            render_container.add_child(component)
                            has_content = True
        else:
            self._content_text.set_custom_bg_fn(bg_fn)
            self._content_text.set_text(self._format_tool_execution())
            has_content = True

        for img in self._image_components:
            self.remove_child(img)
        self._image_components = []
        for spacer in self._image_spacers:
            self.remove_child(spacer)
        self._image_spacers = []

        if self._result:
            image_blocks = [c for c in self._result["content"] if _block_type(c) == "image"]
            caps = get_capabilities()
            for i, img in enumerate(image_blocks):
                if caps["images"] and self._show_images and _block_get(img, "data") and _block_get(img, "mimeType"):
                    converted = self._converted_images.get(i)
                    image_data = converted["data"] if converted else _block_get(img, "data")
                    image_mime_type = converted["mimeType"] if converted else _block_get(img, "mimeType")
                    if caps["images"] == "kitty" and image_mime_type != "image/png":
                        continue

                    spacer = Spacer(1)
                    self.add_child(spacer)
                    self._image_spacers.append(spacer)
                    image_component = Image(
                        image_data,
                        image_mime_type,
                        {"fallbackColor": lambda s: theme.fg("toolOutput", s)},
                        {"maxWidthCells": self._image_width_cells},
                    )
                    self._image_components.append(image_component)
                    self.add_child(image_component)

        if self._has_renderer_definition() and not has_content and not self._image_components:
            self._hide_component = True

    def _get_text_output(self) -> str:
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        from ....core.tools.render_utils import get_text_output

        return get_text_output(self._result, self._show_images)

    def _format_tool_execution(self) -> str:

        text = theme.fg("toolTitle", theme.bold(self._tool_name))
        content = json.dumps(self._args, indent=2)
        if content:
            text += f"\n\n{content}"
        output = self._get_text_output()
        if output:
            text += f"\n{output}"
        return text
