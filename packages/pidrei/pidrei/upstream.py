"""What pi commit this build is a port of.

The runtime source of truth lives here, inside the package, because the wheel is
what has to answer `pidrei --version` on a user's machine — the repo-root
`.last_upstream_ref` is not shipped. That file stays as the copy tooling reads
without importing pidrei (Makefile, CI, `make upstream-diff` when it lands), and
`test_version_scheme.py` asserts the two agree so they cannot drift.

`UPSTREAM_VERSION` is also load-bearing for the distribution version: pidrei is
versioned `<pi version>.<our build>`, so the first three segments of every
package's version must equal this. That is asserted too — the pi ref and the
version number cannot be bumped independently.
"""

#: Upstream project this is a port of.
UPSTREAM_REPO = "https://github.com/earendil-works/pi"

#: pi's released version at UPSTREAM_REF.
UPSTREAM_VERSION = "0.82.0"

#: The exact pi commit ported. `7df73a00` is the commit immediately after the
#: v0.82.0 release ("Add [Unreleased] section for next cycle").
UPSTREAM_REF = "7df73a00c6cf85c000bf1ce1594c9284067a92f0"


def short_ref() -> str:
    return UPSTREAM_REF[:8]
