"""Mirror of pi coding-agent src/modes/interactive/components/index.ts.

UI Components for extensions.
"""

from .armin import ArminComponent
from .assistant_message import AssistantMessageComponent
from .bash_execution import BashExecutionComponent
from .bordered_loader import BorderedLoader
from .branch_summary_message import BranchSummaryMessageComponent
from .compaction_summary_message import CompactionSummaryMessageComponent
from .config_selector import ConfigSelectorComponent
from .countdown_timer import CountdownTimer
from .custom_editor import CustomEditor
from .custom_entry import CustomEntryComponent
from .custom_message import CustomMessageComponent
from .daxnuts import DaxnutsComponent
from .diff import render_diff
from .dynamic_border import DynamicBorder
from .earendil_announcement import EarendilAnnouncementComponent
from .extension_editor import ExtensionEditorComponent
from .extension_input import ExtensionInputComponent
from .extension_selector import ExtensionSelectorComponent
from .first_time_setup import FirstTimeSetupComponent
from .footer import FooterComponent, format_cwd_for_footer, format_tokens
from .keybinding_hints import format_key_text, key_display_text, key_hint, key_text, raw_key_hint
from .login_dialog import LoginDialogComponent
from .model_selector import ModelSelectorComponent
from .oauth_selector import OAuthSelectorComponent, format_auth_selector_provider_type
from .scoped_models_selector import ScopedModelsSelectorComponent
from .session_selector import SessionSelectorComponent
from .settings_selector import SettingsSelectorComponent
from .show_images_selector import ShowImagesSelectorComponent
from .skill_invocation_message import SkillInvocationMessageComponent
from .status_indicator import (
    BranchSummaryStatusIndicator,
    CompactionStatusIndicator,
    IdleStatus,
    RetryStatusIndicator,
    StatusIndicator,
    WorkingStatusIndicator,
)
from .theme_selector import ThemeSelectorComponent
from .thinking_selector import ThinkingSelectorComponent
from .tool_execution import ToolExecutionComponent
from .tree_selector import TreeSelectorComponent
from .trust_selector import TrustSelectorComponent
from .user_message import UserMessageComponent
from .user_message_selector import UserMessageSelectorComponent
from .visual_truncate import truncate_to_visual_lines


__all__ = [
    "ArminComponent",
    "AssistantMessageComponent",
    "BashExecutionComponent",
    "BorderedLoader",
    "BranchSummaryMessageComponent",
    "BranchSummaryStatusIndicator",
    "CompactionStatusIndicator",
    "CompactionSummaryMessageComponent",
    "ConfigSelectorComponent",
    "CountdownTimer",
    "CustomEditor",
    "CustomEntryComponent",
    "CustomMessageComponent",
    "DaxnutsComponent",
    "DynamicBorder",
    "EarendilAnnouncementComponent",
    "ExtensionEditorComponent",
    "ExtensionInputComponent",
    "ExtensionSelectorComponent",
    "FirstTimeSetupComponent",
    "FooterComponent",
    "IdleStatus",
    "LoginDialogComponent",
    "ModelSelectorComponent",
    "OAuthSelectorComponent",
    "RetryStatusIndicator",
    "ScopedModelsSelectorComponent",
    "SessionSelectorComponent",
    "SettingsSelectorComponent",
    "ShowImagesSelectorComponent",
    "SkillInvocationMessageComponent",
    "StatusIndicator",
    "ThemeSelectorComponent",
    "ThinkingSelectorComponent",
    "ToolExecutionComponent",
    "TreeSelectorComponent",
    "TrustSelectorComponent",
    "UserMessageComponent",
    "UserMessageSelectorComponent",
    "WorkingStatusIndicator",
    "format_auth_selector_provider_type",
    "format_cwd_for_footer",
    "format_key_text",
    "format_tokens",
    "key_display_text",
    "key_hint",
    "key_text",
    "raw_key_hint",
    "render_diff",
    "truncate_to_visual_lines",
]
