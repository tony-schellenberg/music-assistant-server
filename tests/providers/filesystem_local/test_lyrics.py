"""Tests for lyrics loaded from Local files."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from music_assistant.helpers.tags import AudioTags
from tests.providers.filesystem_local.conftest import make_provider

if TYPE_CHECKING:
    from pathlib import Path

    from music_assistant.mass import MusicAssistant

TRACK_FILE = "01 - Come Together.flac"


def _make_tags(lyrics: str) -> AudioTags:
    """Build the tags of a track with embedded lyrics."""
    return AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=1000,
        duration=259.0,
        tags={"title": "Come Together", "artist": "The Beatles", "track": "1", "lyrics": lyrics},
        has_cover_image=False,
        filename=TRACK_FILE,
    )


@pytest.mark.parametrize(
    "sidecar_lyrics",
    ["plain sidecar lyrics", "[00:01.00]synchronized sidecar lyrics"],
    ids=["plain-sidecar", "synchronized-sidecar"],
)
async def test_lrc_sidecar_is_preserved_alongside_embedded_lyrics(
    mass_minimal: MusicAssistant, tmp_path: Path, sidecar_lyrics: str
) -> None:
    """A sidecar overrides embedded lyrics even when it contains no timestamps."""
    (tmp_path / TRACK_FILE).write_bytes(b"\x00" * 128)
    (tmp_path / "01 - Come Together.lrc").write_text(sidecar_lyrics, encoding="utf-8")
    provider = make_provider(mass_minimal, tmp_path)
    provider._resolve_artists_with_mbids = AsyncMock(return_value=[])  # type: ignore[method-assign]
    file_item = await provider.resolve(TRACK_FILE)

    tags = _make_tags("embedded tag lyrics")
    tags.tags["synchronizedlyrics"] = [("embedded synchronized lyrics", 1000)]
    track = await provider._parse_track(file_item, tags)

    if sidecar_lyrics.startswith("["):
        assert track.metadata.lyrics == "embedded tag lyrics"
        assert track.metadata.lrc_lyrics == sidecar_lyrics
    else:
        assert track.metadata.lyrics == sidecar_lyrics
        assert track.metadata.lrc_lyrics is None


async def test_embedded_lyrics_remain_when_no_lrc_sidecar_exists(
    mass_minimal: MusicAssistant, tmp_path: Path
) -> None:
    """Embedded tag lyrics remain available when there is no sidecar file."""
    (tmp_path / TRACK_FILE).write_bytes(b"\x00" * 128)
    provider = make_provider(mass_minimal, tmp_path)
    provider._resolve_artists_with_mbids = AsyncMock(return_value=[])  # type: ignore[method-assign]
    file_item = await provider.resolve(TRACK_FILE)

    track = await provider._parse_track(file_item, _make_tags("embedded tag lyrics"))

    assert track.metadata.lyrics == "embedded tag lyrics"
    assert track.metadata.lrc_lyrics is None
