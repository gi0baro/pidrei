"""pidrei-ai: model types, the provider registry, wire adapters and auth."""


def __getattr__(name: str) -> str:
    """`__version__` on demand (PEP 562).

    Derived from installed metadata rather than written here, because a literal
    is one more place a release has to remember — this one sat at `0.1.0.dev0`
    through two version bumps before anything noticed. Lazy because
    `importlib.metadata.version` walks distribution metadata on disk, and this
    module is imported on every start while `__version__` is read by almost
    nobody.
    """
    if name == "__version__":
        import importlib.metadata

        try:
            return importlib.metadata.version("pidrei-ai")
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
