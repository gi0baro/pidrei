"""Mirror of pi coding-agent test/trust-manager.test.ts."""

import pytest

from pidrei.core.trust_manager import ProjectTrustStore, has_trust_requiring_project_resources


@pytest.mark.tonio
async def test_stores_decisions_and_inherits_from_parent_directories(tmp_dir):
    agent_dir = tmp_dir / "agent"
    agent_dir.mkdir()
    store = ProjectTrustStore(str(agent_dir))
    parent_dir = tmp_dir / "trusted-parent"
    child_dir = parent_dir / "project"
    child_dir.mkdir(parents=True)

    assert await store.get(str(child_dir)) is None
    await store.set(str(parent_dir), True)
    assert await store.get(str(child_dir)) is True
    await store.set(str(child_dir), False)
    assert await store.get(str(child_dir)) is False
    await store.set(str(child_dir), None)
    assert await store.get(str(child_dir)) is True


def test_detects_trust_requiring_project_resources(tmp_dir, monkeypatch):
    cwd = tmp_dir / "project"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(tmp_dir))

    (tmp_dir / ".pidrei" / "agent").mkdir(parents=True)
    (tmp_dir / ".agents" / "skills").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(tmp_dir)) is False
    assert has_trust_requiring_project_resources(str(cwd)) is False

    (tmp_dir / ".pidrei" / "settings.json").write_text("{}", encoding="utf-8")
    assert has_trust_requiring_project_resources(str(tmp_dir)) is True
    (tmp_dir / ".pidrei" / "settings.json").unlink()

    (cwd / ".pidrei").mkdir()
    (cwd / ".pidrei" / "settings.json").write_text("{}", encoding="utf-8")
    assert has_trust_requiring_project_resources(str(cwd)) is True

    (cwd / ".pidrei" / "settings.json").unlink()
    (cwd / ".pidrei").rmdir()
    (cwd / ".agents" / "skills").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(cwd)) is True
