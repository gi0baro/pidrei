"""Mirror of pi coding-agent src/cli/file-processor.ts.

Process @file CLI arguments into text content and image attachments.
"""

import os
import sys
from dataclasses import dataclass, field

from pidrei_ai.types import ImageContent

from ..core.tools.path_utils import resolve_read_path
from ..utils.colors import red
from ..utils.image_process import process_image
from ..utils.mime import detect_supported_image_mime_type_from_file


@dataclass(slots=True)
class ProcessedFiles:
    text: str = ""
    images: list[ImageContent] = field(default_factory=list)


def process_file_arguments(file_args: list[str], *, auto_resize_images: bool = True) -> ProcessedFiles:
    """Process @file arguments into text content and image attachments."""
    text = ""
    images: list[ImageContent] = []

    for file_arg in file_args:
        # Expand and resolve path (handles ~ expansion and macOS screenshot Unicode spaces)
        absolute_path = os.path.abspath(resolve_read_path(file_arg, os.getcwd()))

        # Check if file exists
        if not os.path.exists(absolute_path):
            print(red(f"Error: File not found: {absolute_path}"), file=sys.stderr)
            raise SystemExit(1)

        # Check if file is empty
        if os.path.getsize(absolute_path) == 0:
            # Skip empty files
            continue

        mime_type = detect_supported_image_mime_type_from_file(absolute_path)

        if mime_type:
            # Handle image file
            with open(absolute_path, "rb") as handle:
                content = handle.read()
            processed = process_image(content, mime_type, auto_resize_images=auto_resize_images)

            if not processed.ok:
                text += f'<file name="{absolute_path}">{processed.message}</file>\n'
                continue

            images.append(ImageContent(mime_type=processed.mime_type, data=processed.data))

            # Add text reference to image with optional processing hints
            if processed.hints:
                hints = "\n".join(processed.hints)
                text += f'<file name="{absolute_path}">{hints}</file>\n'
            else:
                text += f'<file name="{absolute_path}"></file>\n'
        else:
            # Handle text file
            try:
                with open(absolute_path, encoding="utf-8") as handle:
                    file_content = handle.read()
                text += f'<file name="{absolute_path}">\n{file_content}\n</file>\n'
            except Exception as error:
                print(red(f"Error: Could not read file {absolute_path}: {error}"), file=sys.stderr)
                raise SystemExit(1) from None

    return ProcessedFiles(text=text, images=images)
