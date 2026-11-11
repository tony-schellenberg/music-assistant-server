"""Tests for the on-demand lyrics lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ProviderFeature
from music_assistant_models.media_items import ProviderMapping
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.controllers.music import MusicController
from music_assistant.models.music_provider import MusicProvider
from tests.controllers.music.helpers import create_track

if TYPE_CHECKING:
    from music_assistant.controllers.metadata import MetaDataController


@pytest.mark.parametrize(
    ("library", "local_lyrics", "local_lrc", "cached_lyrics", "cached_lrc"),
    [
        (False, "local tags", None, None, "[00:01.00]cached lyrics"),
        (False, None, "[00:01.00]local sidecar", "cached tags", None),
        (True, "local tags", None, None, "[00:01.00]cached lyrics"),
        (True, None, "[00:01.00]local sidecar", "cached tags", None),
    ],
)
async def test_filesystem_lyrics_override_cached_lyrics(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    library: bool,
    local_lyrics: str | None,
    local_lrc: str | None,
    cached_lyrics: str | None,
    cached_lrc: str | None,
) -> None:
    """Local tags or sidecars are not replaced by cached or remote lyrics."""
    mass = metadata_controller.mass
    mass.music = MusicController(mass)
    filesystem = MagicMock(spec=MusicProvider)
    filesystem.domain = "filesystem_local"
    filesystem.supported_features = set()
    monkeypatch.setattr(mass.music, "get_visible_provider", MagicMock(return_value=filesystem))

    remote = MagicMock()
    remote.supported_features = {ProviderFeature.LYRICS}
    remote.is_local = False
    remote.name = "remote"
    remote.get_track_metadata = AsyncMock(
        return_value=MediaItemMetadata(lrc_lyrics="[00:02.00]remote synchronized")
    )
    monkeypatch.setattr(mass, "get_providers", MagicMock(return_value=[remote]))

    if library:
        track = create_track("library", "library-track")
        track.provider = "library"
        track.provider_mappings = {
            ProviderMapping(
                item_id="local-track",
                provider_domain="filesystem_local",
                provider_instance="local_1",
            )
        }
        monkeypatch.setattr(mass.music.tracks, "get", AsyncMock(return_value=track))
        monkeypatch.setattr(metadata_controller, "_update_track_metadata", AsyncMock())
    else:
        track = create_track("local_1", "local-track")

    track.metadata = MediaItemMetadata(lyrics=cached_lyrics, lrc_lyrics=cached_lrc)
    local_track = create_track("local_1", "local-track")
    local_track.metadata = MediaItemMetadata(lyrics=local_lyrics, lrc_lyrics=local_lrc)
    monkeypatch.setattr(mass.music.tracks, "get_provider_item", AsyncMock(return_value=local_track))

    assert await metadata_controller.get_track_lyrics(track) == (local_lyrics, local_lrc)
    remote.get_track_metadata.assert_not_awaited()


async def test_first_filesystem_lyrics_stop_lookup(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain local lyrics stop lookup without searching other mappings or remote sources."""
    mass = metadata_controller.mass
    mass.music = MusicController(mass)
    tags_provider = MagicMock(spec=MusicProvider)
    tags_provider.domain = "filesystem_local"
    sidecar_provider = MagicMock(spec=MusicProvider)
    sidecar_provider.domain = "filesystem_sidecar"
    providers = {"tags": tags_provider, "sidecar": sidecar_provider}
    monkeypatch.setattr(
        mass.music,
        "get_visible_provider",
        MagicMock(side_effect=lambda instance_id: providers[instance_id]),
    )

    track = create_track("library", "library-track")
    track.provider = "library"
    track.provider_mappings = {
        ProviderMapping(
            item_id="tags-track",
            provider_domain="filesystem_local",
            provider_instance="tags",
        ),
        ProviderMapping(
            item_id="sidecar-track",
            provider_domain="filesystem_sidecar",
            provider_instance="sidecar",
        ),
    }
    monkeypatch.setattr(mass.music.tracks, "get", AsyncMock(return_value=track))
    monkeypatch.setattr(metadata_controller, "_update_track_metadata", AsyncMock())
    tags_track = create_track("tags", "tags-track")
    tags_track.metadata.lyrics = "local tags"
    sidecar_track = create_track("sidecar", "sidecar-track")
    sidecar_track.metadata.lrc_lyrics = "[00:01.00]local sidecar"
    get_provider_item = AsyncMock(side_effect=[tags_track, sidecar_track])
    monkeypatch.setattr(mass.music.tracks, "get_provider_item", get_provider_item)
    search_remote_lyrics = AsyncMock()
    monkeypatch.setattr(metadata_controller, "_search_remote_lyrics", search_remote_lyrics)

    assert await metadata_controller.get_track_lyrics(track) == ("local tags", None)
    assert get_provider_item.await_count == 1
    search_remote_lyrics.assert_not_awaited()


async def test_invisible_filesystem_mapping_is_not_used_for_lyrics(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inaccessible filesystem source cannot supply a lyrics override."""
    mass = metadata_controller.mass
    mass.music = MusicController(mass)
    monkeypatch.setattr(mass.music, "get_visible_provider", MagicMock(return_value=None))
    track = create_track("library", "library-track")
    track.provider = "library"
    track.provider_mappings = {
        ProviderMapping(
            item_id="local-track",
            provider_domain="filesystem_local",
            provider_instance="local_1",
        )
    }
    monkeypatch.setattr(mass.music.tracks, "get", AsyncMock(return_value=track))
    monkeypatch.setattr(metadata_controller, "_update_track_metadata", AsyncMock())
    get_provider_item = AsyncMock(return_value=create_track("local_1", "local-track"))
    monkeypatch.setattr(mass.music.tracks, "get_provider_item", get_provider_item)
    monkeypatch.setattr(mass, "get_providers", MagicMock(return_value=[]))

    assert await metadata_controller.get_track_lyrics(track) == (None, None)
    get_provider_item.assert_not_awaited()


@pytest.mark.parametrize(
    (
        "provider_domain",
        "cached_lyrics",
        "provider_item",
        "item_lyrics",
        "expected_lyrics",
        "expected_local_override",
        "expected_fetches",
    ),
    [
        (
            "filesystem_local",
            None,
            "local_1",
            "file lyrics",
            "file lyrics",
            True,
            1,
        ),
        (
            "filesystem_local",
            "cached lyrics",
            "library",
            "stale library lyrics",
            "cached lyrics",
            True,
            1,
        ),
        (
            "filesystem_local",
            None,
            "library",
            "stale library lyrics",
            None,
            False,
            1,
        ),
        (
            "spotify",
            None,
            "spotify",
            "provider lyrics",
            "provider lyrics",
            False,
            1,
        ),
        (
            "spotify",
            "cached lyrics",
            "spotify",
            "ignored",
            "cached lyrics",
            False,
            0,
        ),
        ("spotify", None, "spotify", None, None, False, 1),
    ],
)
async def test_direct_provider_lyrics_lookup(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    provider_domain: str,
    cached_lyrics: str | None,
    provider_item: str,
    item_lyrics: str | None,
    expected_lyrics: str | None,
    expected_local_override: bool,
    expected_fetches: int,
) -> None:
    """Direct lookup fetches filesystem lyrics always and other providers on demand."""
    mass = metadata_controller.mass
    mass.music = MusicController(mass)
    provider = MagicMock(spec=MusicProvider)
    provider.domain = provider_domain
    provider.supported_features = (
        set() if provider_domain.startswith("filesystem_") else {ProviderFeature.LYRICS}
    )
    monkeypatch.setattr(mass.music, "get_visible_provider", MagicMock(return_value=provider))

    provider_instance = "local_1" if provider_domain.startswith("filesystem_") else "spotify"
    track = create_track(provider_instance, "t1")
    track.metadata = MediaItemMetadata(lyrics=cached_lyrics)
    full_track = create_track(provider_item, "t1")
    full_track.metadata = MediaItemMetadata(lyrics=item_lyrics)
    get_provider_item = AsyncMock(return_value=full_track)
    monkeypatch.setattr(mass.music.tracks, "get_provider_item", get_provider_item)

    result = await metadata_controller._get_track_lyrics(track)

    assert result == (track, expected_lyrics, None, expected_local_override)
    assert get_provider_item.await_count == expected_fetches
