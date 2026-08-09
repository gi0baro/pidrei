"""Mirror of pi coding-agent src/cli/args.ts.

CLI argument parsing and help display. The parser is a deliberate hand-port
of pi's single-pass loop: unknown `--flags` are collected (extensions may
register CLI flags validated only after discovery), errors accumulate as
diagnostics instead of aborting, and `@file`/message positionals may appear
anywhere.
"""

from dataclasses import dataclass, field
from typing import Any

from ..config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, ENV_SESSION_DIR
from ..utils.colors import bold


# Mode = "text" | "json" | "rpc"

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def is_valid_thinking_level(value: str) -> bool:
    return value in THINKING_LEVELS


@dataclass(slots=True)
class Args:
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    append_system_prompt: list[str] | None = None
    thinking: str | None = None
    continue_: bool | None = None
    resume: bool | None = None
    help: bool | None = None
    version: bool | None = None
    mode: str | None = None
    name: str | None = None
    no_session: bool | None = None
    session: str | None = None
    session_id: str | None = None
    fork: str | None = None
    session_dir: str | None = None
    models: list[str] | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: bool | None = None
    no_builtin_tools: bool | None = None
    extensions: list[str] | None = None
    no_extensions: bool | None = None
    print: bool | None = None
    export: str | None = None
    no_skills: bool | None = None
    skills: list[str] | None = None
    prompt_templates: list[str] | None = None
    no_prompt_templates: bool | None = None
    themes: list[str] | None = None
    no_themes: bool | None = None
    no_context_files: bool | None = None
    # str for a search pattern, True for a bare --list-models
    list_models: str | bool | None = None
    offline: bool | None = None
    alt: bool | None = None
    verbose: bool | None = None
    project_trust_override: bool | None = None
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)
    # Unknown flags (potentially extension flags) - map of flag name to value
    unknown_flags: dict[str, bool | str] = field(default_factory=dict)
    diagnostics: list[dict[str, str]] = field(default_factory=list)


def parse_args(args: list[str]) -> Args:  # noqa: C901
    result = Args()

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and i + 1 < len(args):
            i += 1
            mode = args[i]
            if mode in ("text", "json", "rpc"):
                result.mode = mode
        elif arg in ("--continue", "-c"):
            result.continue_ = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and i + 1 < len(args):
            i += 1
            result.provider = args[i]
        elif arg == "--model" and i + 1 < len(args):
            i += 1
            result.model = args[i]
        elif arg == "--api-key" and i + 1 < len(args):
            i += 1
            result.api_key = args[i]
        elif arg == "--system-prompt" and i + 1 < len(args):
            i += 1
            result.system_prompt = args[i]
        elif arg == "--append-system-prompt" and i + 1 < len(args):
            i += 1
            result.append_system_prompt = result.append_system_prompt or []
            result.append_system_prompt.append(args[i])
        elif arg in ("--name", "-n"):
            if i + 1 < len(args):
                i += 1
                result.name = args[i]
            else:
                result.diagnostics.append({"type": "error", "message": "--name requires a value"})
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and i + 1 < len(args):
            i += 1
            result.session = args[i]
        elif arg == "--session-id" and i + 1 < len(args):
            i += 1
            result.session_id = args[i]
        elif arg == "--fork" and i + 1 < len(args):
            i += 1
            result.fork = args[i]
        elif arg == "--session-dir" and i + 1 < len(args):
            i += 1
            result.session_dir = args[i]
        elif arg == "--models" and i + 1 < len(args):
            i += 1
            result.models = [s.strip() for s in args[i].split(",")]
        elif arg in ("--no-tools", "-nt"):
            result.no_tools = True
        elif arg in ("--no-builtin-tools", "-nbt"):
            result.no_builtin_tools = True
        elif arg in ("--tools", "-t") and i + 1 < len(args):
            i += 1
            result.tools = [name for name in (s.strip() for s in args[i].split(",")) if name]
        elif arg in ("--exclude-tools", "-xt") and i + 1 < len(args):
            i += 1
            result.exclude_tools = [name for name in (s.strip() for s in args[i].split(",")) if name]
        elif arg == "--thinking" and i + 1 < len(args):
            i += 1
            level = args[i]
            if is_valid_thinking_level(level):
                result.thinking = level
            else:
                result.diagnostics.append(
                    {
                        "type": "warning",
                        "message": f'Invalid thinking level "{level}". Valid values: {", ".join(THINKING_LEVELS)}',
                    }
                )
        elif arg in ("--print", "-p"):
            result.print = True
            nxt = args[i + 1] if i + 1 < len(args) else None
            if nxt is not None and not nxt.startswith("@") and (not nxt.startswith("-") or nxt.startswith("---")):
                result.messages.append(nxt)
                i += 1
        elif arg == "--export" and i + 1 < len(args):
            i += 1
            result.export = args[i]
        elif arg in ("--extension", "-e") and i + 1 < len(args):
            i += 1
            result.extensions = result.extensions or []
            result.extensions.append(args[i])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and i + 1 < len(args):
            i += 1
            result.skills = result.skills or []
            result.skills.append(args[i])
        elif arg == "--prompt-template" and i + 1 < len(args):
            i += 1
            result.prompt_templates = result.prompt_templates or []
            result.prompt_templates.append(args[i])
        elif arg == "--theme" and i + 1 < len(args):
            i += 1
            result.themes = result.themes or []
            result.themes.append(args[i])
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg in ("--no-context-files", "-nc"):
            result.no_context_files = True
        elif arg == "--list-models":
            # Check if next arg is a search pattern (not a flag or file arg)
            if i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@"):
                i += 1
                result.list_models = args[i]
            else:
                result.list_models = True
        elif arg == "--alt":
            result.alt = True
        elif arg == "--verbose":
            result.verbose = True
        elif arg in ("--approve", "-a"):
            result.project_trust_override = True
        elif arg in ("--no-approve", "-na"):
            result.project_trust_override = False
        elif arg == "--offline":
            result.offline = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])  # Remove @ prefix
        elif arg.startswith("--"):
            eq_index = arg.find("=")
            if eq_index != -1:
                result.unknown_flags[arg[2:eq_index]] = arg[eq_index + 1 :]
            else:
                flag_name = arg[2:]
                nxt = args[i + 1] if i + 1 < len(args) else None
                if nxt is not None and not nxt.startswith("-") and not nxt.startswith("@"):
                    result.unknown_flags[flag_name] = nxt
                    i += 1
                else:
                    result.unknown_flags[flag_name] = True
        elif arg.startswith("-") and not arg.startswith("--"):
            result.diagnostics.append({"type": "error", "message": f"Unknown option: {arg}"})
        elif not arg.startswith("-"):
            result.messages.append(arg)

        i += 1

    return result


def print_help(extension_flags: list[Any] | None = None) -> None:
    if extension_flags:
        flag_lines = []
        for flag in extension_flags:
            value = " <value>" if flag.type == "string" else ""
            description = flag.description or f"Registered by {flag.extension_path}"
            flag_lines.append(f"  --{flag.name}{value}".ljust(30) + description)
        extension_flags_text = "\n" + bold("Extension CLI Flags:") + "\n" + "\n".join(flag_lines) + "\n"
    else:
        extension_flags_text = ""

    print(f"""{bold(APP_NAME)} - AI coding assistant with read, bash, edit, write tools

{bold("Usage:")}
  {APP_NAME} [options] [@files...] [messages...]

{bold("Commands:")}
  {APP_NAME} install <source> [-l]     Install extension source and add to settings
  {APP_NAME} remove <source> [-l]      Remove extension source from settings
  {APP_NAME} uninstall <source> [-l]   Alias for remove
  {APP_NAME} update [source]           Update installed extensions (no self-update)
  {APP_NAME} update --models           Refresh model catalogs
  {APP_NAME} list                      List installed extensions from settings
  {APP_NAME} config [-l]               Open TUI to enable/disable package resources (Tab switches scope)
  {APP_NAME} auth <command>            Print credentials or check provider readiness
  {APP_NAME} <command> --help          Show help for install/remove/uninstall/update/list/config/auth

{bold("Options:")}
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path|id>            Use specific session file or partial UUID
  --session-id <id>              Use exact project session ID, creating it if missing
  --fork <path|id>               Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --name, -n <name>              Set session display name
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
                                 Supports globs (anthropic/*, *sonnet*) and fuzzy matching
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-builtin-tools, -nbt       Disable built-in tools by default but keep extension/custom tools enabled
  --tools, -t <tools>            Comma-separated allowlist of tool names to enable
                                 Applies to built-in, extension, and custom tools
  --exclude-tools, -xt <tools>   Comma-separated denylist of tool names to disable
                                 Applies to built-in, extension, and custom tools
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh, max
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery (explicit -e paths still work)
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --no-themes                    Disable theme discovery and loading
  --no-context-files, -nc        Disable AGENTS.md and CLAUDE.md discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup (overrides quietStartup setting)
  --alt                          Use the alternate-screen TUI in interactive mode
  --approve, -a                  Trust project-local files for this run
  --no-approve, -na              Ignore project-local files for this run
  --offline                      Disable startup network operations (same as PIDREI_OFFLINE=1)
  --help, -h                     Show this help
  --version, -v                  Show version number

Extensions can register additional flags (e.g., --plan from plan-mode extension).{extension_flags_text}

{bold("Examples:")}
  # Print a provider API key for an external client
  {APP_NAME} auth print-api-key --provider openai

  # Print an OAuth bearer token for an external client (refreshes if expired)
  {APP_NAME} auth print-bearer-token --provider openai-codex

  # Interactive mode
  {APP_NAME}

  # Interactive mode with initial prompt
  {APP_NAME} "List all .py files in src/"

  # Include files in initial message
  {APP_NAME} @prompt.md @image.png "What color is the sky?"

  # Non-interactive mode (process and exit)
  {APP_NAME} -p "List all .py files in src/"

  # Multiple messages (interactive)
  {APP_NAME} "Read pyproject.toml" "What dependencies do we have?"

  # Continue previous session
  {APP_NAME} --continue "What did we discuss?"

  # Start a named session
  {APP_NAME} --name "Refactor auth module"

  # Use different model
  {APP_NAME} --provider openai --model gpt-4o-mini "Help me refactor this code"

  # Use model with provider prefix (no --provider needed)
  {APP_NAME} --model openai/gpt-4o "Help me refactor this code"

  # Use model with thinking level shorthand
  {APP_NAME} --model sonnet:high "Solve this complex problem"

  # Limit model cycling to specific models
  {APP_NAME} --models claude-sonnet,claude-haiku,gpt-4o

  # Limit to a specific provider with glob pattern
  {APP_NAME} --models "github-copilot/*"

  # Cycle models with fixed thinking levels
  {APP_NAME} --models sonnet:high,haiku:low

  # Start with a specific thinking level
  {APP_NAME} --thinking high "Solve this complex problem"

  # Read-only mode (no file modifications possible)
  {APP_NAME} --tools read,grep,find,ls -p "Review the code in src/"

  # Disable one tool while keeping the rest available
  {APP_NAME} --exclude-tools ask_question

  # Export a session file to HTML
  {APP_NAME} --export ~/{CONFIG_DIR_NAME}/agent/sessions/--path--/session.jsonl
  {APP_NAME} --export session.jsonl output.html

{bold("Environment Variables:")}
  ANTHROPIC_AUTH_TOKEN             - Anthropic bearer auth token
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  ANTHROPIC_OAUTH_TOKEN            - Anthropic OAuth token (alternative to API key)
  ANT_LING_API_KEY                 - Ant Ling API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  AZURE_OPENAI_API_KEY             - Azure OpenAI API key
  AZURE_OPENAI_BASE_URL            - Azure OpenAI/Cognitive Services base URL (e.g. https://{{resource}}.openai.azure.com)
  AZURE_OPENAI_RESOURCE_NAME       - Azure OpenAI resource name (alternative to base URL)
  AZURE_OPENAI_API_VERSION         - Azure OpenAI API version (default: v1)
  AZURE_OPENAI_DEPLOYMENT_NAME_MAP - Azure OpenAI model=deployment map (comma-separated)
  DEEPSEEK_API_KEY                 - DeepSeek API key
  NVIDIA_API_KEY                   - NVIDIA NIM API key
  GEMINI_API_KEY                   - Google Gemini API key
  GROQ_API_KEY                     - Groq API key
  CEREBRAS_API_KEY                 - Cerebras API key
  XAI_API_KEY                      - xAI Grok API key
  FIREWORKS_API_KEY                - Fireworks API key
  TOGETHER_API_KEY                 - Together AI API key
  BASETEN_API_KEY                  - Baseten API key
  OPENROUTER_API_KEY               - OpenRouter API key
  AI_GATEWAY_API_KEY               - Vercel AI Gateway API key
  ZAI_API_KEY                      - ZAI Coding Plan API key (Global)
  ZAI_CODING_CN_API_KEY            - ZAI Coding Plan API key (China)
  MISTRAL_API_KEY                  - Mistral API key
  MINIMAX_API_KEY                  - MiniMax API key
  MOONSHOT_API_KEY                 - Moonshot AI API key
  OPENCODE_API_KEY                 - OpenCode Zen/OpenCode Go API key
  KIMI_API_KEY                     - Kimi For Coding API key
  CLOUDFLARE_API_KEY               - Cloudflare API token (Workers AI and AI Gateway)
  CLOUDFLARE_ACCOUNT_ID            - Cloudflare account id (required for both)
  CLOUDFLARE_GATEWAY_ID            - Cloudflare AI Gateway slug (required for AI Gateway)
  QWEN_TOKEN_PLAN_API_KEY          - Qwen Token Plan API key (international region)
  QWEN_TOKEN_PLAN_CN_API_KEY       - Qwen Token Plan API key (China region)
  XIAOMI_API_KEY                   - Xiaomi MiMo API key (api.xiaomimimo.com billing)
  XIAOMI_TOKEN_PLAN_CN_API_KEY     - Xiaomi MiMo Token Plan API key (China region)
  XIAOMI_TOKEN_PLAN_AMS_API_KEY    - Xiaomi MiMo Token Plan API key (Amsterdam region)
  XIAOMI_TOKEN_PLAN_SGP_API_KEY    - Xiaomi MiMo Token Plan API key (Singapore region)
  AWS_PROFILE                      - AWS profile for Amazon Bedrock
  AWS_ACCESS_KEY_ID                - AWS access key for Amazon Bedrock
  AWS_SECRET_ACCESS_KEY            - AWS secret key for Amazon Bedrock
  AWS_BEARER_TOKEN_BEDROCK         - Bedrock API key (bearer token)
  AWS_REGION                       - AWS region for Amazon Bedrock (e.g., us-east-1)
  {ENV_AGENT_DIR.ljust(32)} - Config directory (default: ~/{CONFIG_DIR_NAME}/agent)
  {ENV_SESSION_DIR.ljust(32)} - Session storage directory (overridden by --session-dir)
  PIDREI_PACKAGE_DIR               - Override package directory (for Nix/Guix store paths)
  PIDREI_OFFLINE                   - Disable startup network operations when set to 1/true/yes
  PIDREI_PROVIDER_ATTRIBUTION      - Override provider attribution headers when set to 1/true/yes or 0/false/no
  PIDREI_SHARE_VIEWER_URL          - Base URL of a session viewer for /share (default: none, print the gist URL)

{bold("Built-in Tool Names:")}
  read   - Read file contents
  bash   - Execute bash commands
  edit   - Edit files with find/replace
  write  - Write files (creates/overwrites)
  grep   - Search file contents (read-only, off by default)
  find   - Find files by glob pattern (read-only, off by default)
  ls     - List directory contents (read-only, off by default)
""")
