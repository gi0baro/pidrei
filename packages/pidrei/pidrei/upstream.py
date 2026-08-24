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
UPSTREAM_VERSION = "0.84.2"

#: The exact pi commit ported — by convention the commit immediately after the
#: release tag ("Add [Unreleased] section for next cycle"). Bumped by
#: `make upstream-bump` as each upstream delta lands.
UPSTREAM_REF = "6db110e6fab3a18e0e54a6e1aa5a479c1e282924"


def short_ref() -> str:
    return UPSTREAM_REF[:8]
