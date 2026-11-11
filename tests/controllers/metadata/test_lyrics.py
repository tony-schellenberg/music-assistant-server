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


async def test_filesystem_sidecar_precedes_tags_from_other_mapping(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An LRC sidecar from any local source takes precedence over plain local tags."""
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
    monkeypatch.setattr(
        mass.music.tracks,
        "get_provider_item",
        AsyncMock(side_effect=[tags_track, sidecar_track]),
    )

    assert await metadata_controller.get_track_lyrics(track) == (
        None,
        "[00:01.00]local sidecar",
    )


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
