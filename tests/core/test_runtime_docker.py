from __future__ import annotations

import asyncio
from types import SimpleNamespace

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.docker import DockerSession


class FakeContainer:
    def __init__(self) -> None:
        self.id = "abcdef1234567890"
        self.stopped = False
        self.removed = False

    def stop(self, timeout: int = 10) -> None:
        self.stopped = True

    def remove(self, force: bool = False) -> None:
        self.removed = True


class FakeImages:
    def __init__(self) -> None:
        self.removed: list[tuple[str, bool]] = []

    def remove(self, image: str, force: bool = False) -> None:
        self.removed.append((image, force))


def test_docker_session_close_removes_image_when_enabled():
    container = FakeContainer()
    images = FakeImages()
    client = SimpleNamespace(images=images)
    config = RuntimeConfig(docker={"remove_image_after_use": True})
    session = DockerSession(container, config, client, "repo/image:tag")

    asyncio.run(session.close())

    assert container.stopped is True
    assert container.removed is True
    assert images.removed == [("repo/image:tag", True)]


def test_docker_session_close_keeps_image_when_disabled():
    container = FakeContainer()
    images = FakeImages()
    client = SimpleNamespace(images=images)
    config = RuntimeConfig(docker={"remove_image_after_use": False})
    session = DockerSession(container, config, client, "repo/image:tag")

    asyncio.run(session.close())

    assert container.stopped is True
    assert container.removed is True
    assert images.removed == []
