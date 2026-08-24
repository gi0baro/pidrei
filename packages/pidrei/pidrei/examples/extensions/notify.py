"""Terminal Notify

Sends a native terminal notification when the agent is done and waiting for
input. Supports multiple terminal protocols:

- OSC 777: Ghostty, iTerm2, WezTerm, rxvt-unicode
- OSC 99: Kitty
- Windows toast: Windows Terminal (WSL)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/notify.py
"""

import os
import sys


def _windows_toast_script(title: str, body: str) -> str:
    type_ = "Windows.UI.Notifications"
    mgr = f"[{type_}.ToastNotificationManager, {type_}, ContentType = WindowsRuntime]"
    template = f"[{type_}.ToastTemplateType]::ToastText01"
    toast = f"[{type_}.ToastNotification]::new($xml)"
    return "; ".join(
        [
            f"{mgr} > $null",
            f"$xml = [{type_}.ToastNotificationManager]::GetTemplateContent({template})",
            f"$xml.GetElementsByTagName('text')[0].AppendChild($xml.CreateTextNode('{body}')) > $null",
            f"[{type_}.ToastNotificationManager]::CreateToastNotifier('{title}').Show({toast})",
        ]
    )


def _write_escape(sequence: str) -> None:
    # pidrei does not expose a raw escape-sequence writer to extensions (the
    # terminal keeps its writer private; `ctx.ui.set_title` is the only OSC
    # helper), so this goes straight to stdout — the same thing pi does with
    # `process.stdout.write`. OSC sequences are invisible to the renderer, so
    # they do not disturb the TUI.
    sys.stdout.write(sequence)
    sys.stdout.flush()


def _notify_osc777(title: str, body: str) -> None:
    _write_escape(f"\x1b]777;notify;{title};{body}\x07")


def _notify_osc99(title: str, body: str) -> None:
    # Kitty OSC 99: i=notification id, d=0 means not done yet, p=body for second part
    _write_escape(f"\x1b]99;i=1:d=0;{title}\x1b\\")
    _write_escape(f"\x1b]99;i=1:p=body;{body}\x1b\\")


def extension(pi):
    async def notify(title: str, body: str) -> None:
        if os.environ.get("WT_SESSION"):
            await pi.exec("powershell.exe", ["-NoProfile", "-Command", _windows_toast_script(title, body)])
        elif os.environ.get("KITTY_WINDOW_ID"):
            _notify_osc99(title, body)
        else:
            _notify_osc777(title, body)

    # `agent_end` fires after each low-level run; the agent may still retry,
    # compact, or continue with queued follow-ups. Notify only after the full
    # run settles.
    async def on_agent_settled(_event, _ctx) -> None:
        await notify("Pidrei", "Ready for input")

    pi.on("agent_settled", on_agent_settled)
