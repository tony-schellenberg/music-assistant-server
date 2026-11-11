"""Tests for lyrics parsed from local audio files and sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from music_assistant.helpers.tags import AudioTags
from tests.providers.filesystem_local.conftest import make_provider

if TYPE_CHECKING:
    from music_assistant_models.media_items import Track

    from music_assistant.mass import MusicAssistant


def _make_tags(**lyrics_tags: str) -> AudioTags:
    """Build minimal tags for a local audio track."""
    return AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=1000,
        duration=180.0,
        tags={"title": "Song", "artist": "Artist"} | lyrics_tags,
        has_cover_image=False,
        filename="Song.flac",
    )


async def _parse_track(mass_minimal: MusicAssistant, tmp_path: Path, tags: AudioTags) -> Track:
    """Parse a test audio file using the filesystem provider."""
    (tmp_path / "Song.flac").write_bytes(b"audio")
    provider = make_provider(mass_minimal, tmp_path)
    provider._resolve_artists_with_mbids = AsyncMock(return_value=[])  # type: ignore[method-assign]
    file_item = await provider.resolve("Song.flac")
    return await provider._parse_track(file_item, tags)


@pytest.mark.parametrize(
    ("sidecar", "expected_lyrics", "expected_lrc"),
    [
        ("Sidecar plain lyrics", "Sidecar plain lyrics", None),
        (
            "[00:01.00]Sidecar synchronized lyrics",
            "Embedded plain lyrics",
            "[00:01.00]Sidecar synchronized lyrics",
        ),
    ],
)
async def test_sidecar_lyrics_are_classified_and_override_embedded_lyrics(
    mass_minimal: MusicAssistant,
    tmp_path: Path,
    sidecar: str,
    expected_lyrics: str,
    expected_lrc: str | None,
) -> None:
    """A sidecar is preferred, but untimed text remains plain lyrics."""
    (tmp_path / "Song.lrc").write_text(sidecar, encoding="utf-8")

    tags = _make_tags(lyrics="Embedded plain lyrics")
    tags.tags["synchronizedlyrics"] = [("Embedded synchronized lyrics", 1000)]
    track = await _parse_track(mass_minimal, tmp_path, tags)

    assert track.metadata.lyrics == expected_lyrics
    assert track.metadata.lrc_lyrics == expected_lrc


@pytest.mark.parametrize(
    "embedded", ["Plain embedded lyrics", "[00:01.00]First line\n[00:02.00]Second line"]
)
async def test_embedded_lyrics_are_classified(
    mass_minimal: MusicAssistant, tmp_path: Path, embedded: str
) -> None:
    """Embedded text is synchronized only when it contains sufficient timestamps."""
    track = await _parse_track(mass_minimal, tmp_path, _make_tags(lyrics=embedded))

    assert track.metadata.lyrics == embedded
    assert track.metadata.lrc_lyrics == (embedded if embedded.startswith("[00:") else None)


@pytest.mark.parametrize("sidecar", ["", " \n"])
async def test_empty_sidecar_falls_back_to_embedded_lyrics(
    mass_minimal: MusicAssistant, tmp_path: Path, sidecar: str
) -> None:
    """An empty sidecar does not hide timestamped lyrics embedded in the audio file."""
    (tmp_path / "Song.lrc").write_text(sidecar, encoding="utf-8")
    embedded = "[00:01.00]First line\n[00:02.00]Second line"

    track = await _parse_track(mass_minimal, tmp_path, _make_tags(lyrics=embedded))

    assert track.metadata.lyrics == embedded
    assert track.metadata.lrc_lyrics == embedded


async def test_unreadable_sidecar_falls_back_to_embedded_lyrics(
    mass_minimal: MusicAssistant, tmp_path: Path
) -> None:
    """An unreadable sidecar does not hide timestamped lyrics embedded in the audio file."""
    (tmp_path / "Song.flac").write_bytes(b"audio")
    (tmp_path / "Song.lrc").write_text("unreadable", encoding="utf-8")
    embedded = "[00:01.00]First line\n[00:02.00]Second line"

    provider = make_provider(mass_minimal, tmp_path)
    provider._resolve_artists_with_mbids = AsyncMock(return_value=[])  # type: ignore[method-assign]
    provider._read_file = AsyncMock(side_effect=OSError("unreadable"))  # type: ignore[method-assign]
    file_item = await provider.resolve("Song.flac")
    track = await provider._parse_track(file_item, _make_tags(lyrics=embedded))

    assert track.metadata.lyrics == embedded
    assert track.metadata.lrc_lyrics == embedded
