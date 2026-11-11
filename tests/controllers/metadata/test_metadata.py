"""Tests for the MetaDataController public API."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.config_entries import ProviderAccess
from music_assistant_models.enums import ProviderFeature, ProviderSharing, ProviderType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import (
    MediaItemMetadata,
    MediaItemPalette,
    ProviderMapping,
    Track,
)

from music_assistant.controllers.metadata.constants import CONF_PREFER_SYNCED_LYRICS
from music_assistant.controllers.music import MusicController
from music_assistant.controllers.webserver.helpers.auth_middleware import (
    current_user,
    get_current_user,
    impersonated_user,
)
from music_assistant.models.music_provider import MusicProvider
from tests.common import set_music_source_access
from tests.controllers.music.helpers import create_track

if TYPE_CHECKING:
    from music_assistant.controllers.metadata import MetaDataController
    from music_assistant.mass import MusicAssistant

# every test here registers image ids, which are persisted to the cache database
pytestmark = pytest.mark.usefixtures("cache_database")


async def test_get_image_palette_resolves_registered_id(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered image id resolves to its (provider, path) and yields its palette."""
    palette = MediaItemPalette(primary=(1, 2, 3))
    calls: dict[str, object] = {}

    async def fake_get_palette(_mass: object, path: str, provider: str) -> MediaItemPalette:
        calls["get_palette"] = (path, provider)
        return palette

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", fake_get_palette)

    image_id = metadata_controller.compute_image_id("spotify", "cover.jpg")
    assert await metadata_controller.get_image_palette(image_id) is palette
    assert calls["get_palette"] == ("cover.jpg", "spotify")


async def test_get_image_palette_rejects_imageproxy_url(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a bare image id is accepted; a full /imageproxy/<id> URL yields None."""
    fetched = False

    async def fake_get_palette(_mass: object, _path: str, _provider: str) -> MediaItemPalette:
        nonlocal fetched
        fetched = True
        return MediaItemPalette(primary=(4, 5, 6))

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", fake_get_palette)

    image_id = metadata_controller.compute_image_id("filesystem", "/x.jpg")
    url = f"http://mass.local/imageproxy/{image_id}?size=256"
    assert await metadata_controller.get_image_palette(url) is None
    assert fetched is False


async def test_get_image_palette_rejects_unregistered_input(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown ids and arbitrary URLs yield None and never trigger a fetch (no SSRF)."""
    fetched = False

    async def fake_get_palette(_mass: object, _path: str, _provider: str) -> MediaItemPalette:
        nonlocal fetched
        fetched = True
        return MediaItemPalette(primary=(0, 0, 0))

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", fake_get_palette)

    # an id that was never registered
    assert await metadata_controller.get_image_palette("0" * 64) is None
    # an arbitrary URL must not be resolved or fetched
    assert await metadata_controller.get_image_palette("http://169.254.169.254/latest") is None
    assert fetched is False


async def test_get_image_palette_returns_none_for_unreadable_image(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered image whose data can't be read yields None instead of raising."""

    async def boom(_mass: object, path: str, _provider: str) -> MediaItemPalette:
        raise FileNotFoundError(path)

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", boom)
    image_id = metadata_controller.compute_image_id("filesystem", "missing.jpg")
    assert await metadata_controller.get_image_palette(image_id) is None


async def test_update_metadata_runs_as_the_server(
    metadata_controller: MetaDataController,
) -> None:
    """The refresh fills the household's library item from all sources, not the caller's."""
    service = User(user_id="user-service", username="ha", role=UserRole.SERVICE)
    member = User(user_id="user-member", username="member", role=UserRole.USER)
    refreshing_users: list[User | None] = []

    async def _refresh(*_: object, **__: object) -> None:
        refreshing_users.append(get_current_user())

    metadata_controller._update_track_metadata = AsyncMock(side_effect=_refresh)  # type: ignore[method-assign]
    track = create_track("spotify_theirs", "t1")
    track.item_id, track.provider = "1", "library"

    current_user_token = current_user.set(service)
    impersonated_user_token = impersonated_user.set(member)
    try:
        await metadata_controller.update_metadata(track)
        assert refreshing_users == [None]
        assert get_current_user() is member
    finally:
        impersonated_user.reset(impersonated_user_token)
        current_user.reset(current_user_token)


async def test_lyrics_come_only_from_a_source_the_user_may_see(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """A track's own provider is asked for lyrics only when it is one of the user's sources."""
    owner = User(user_id="user-owner", username="owner", role=UserRole.USER)
    member = User(user_id="user-member", username="member", role=UserRole.USER)
    theirs = MagicMock(spec=MusicProvider)
    theirs.instance_id, theirs.domain, theirs.type = "spotify_theirs", "spotify", ProviderType.MUSIC
    theirs.available, theirs.is_streaming_provider = True, True
    theirs.supported_features = {ProviderFeature.LYRICS}
    theirs.initialized = MagicMock(is_set=MagicMock(return_value=True))
    full_track = create_track("spotify_theirs", "t1")
    full_track.metadata.lyrics = "secret"
    theirs.get_track = AsyncMock(return_value=full_track)
    mass_minimal.music = MusicController(mass_minimal)
    mass_minimal._providers = {"spotify_theirs": theirs}
    set_music_source_access(
        mass_minimal,
        {"spotify_theirs": ProviderAccess(owner=owner.user_id, sharing=ProviderSharing.PRIVATE)},
    )
    track = create_track("spotify_theirs", "t1")

    for user, lyrics in ((owner, "secret"), (member, None)):
        with (
            patch(
                "music_assistant.controllers.music.controller.get_current_user", return_value=user
            ),
            patch(
                "music_assistant.controllers.music.media.base.get_current_user", return_value=user
            ),
        ):
            assert (await metadata_controller.get_track_lyrics(track))[0] == lyrics


async def test_lyrics_of_a_library_track_use_the_stored_item_without_refresh(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Library lookup checks access in caller context and does not refresh stored metadata."""
    owner = User(user_id="user-owner", username="owner", role=UserRole.USER)
    member = User(user_id="user-member", username="member", role=UserRole.USER)
    mass_minimal.music = MusicController(mass_minimal)
    mass_minimal.streams = MagicMock()
    mass_minimal.streams.audio_analysis.get_track_audio_metadata = AsyncMock(return_value=None)
    mass_minimal.webserver = MagicMock()
    mass_minimal.webserver.auth.get_user = AsyncMock(return_value=None)
    mass_minimal.webserver.auth.list_users = AsyncMock(return_value=[])
    # the minimal server runs no cache database
    mass_minimal.cache.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    mass_minimal.cache.set = AsyncMock()  # type: ignore[method-assign]
    await mass_minimal.music._setup_database()
    try:
        set_music_source_access(
            mass_minimal,
            {
                "spotify_theirs": ProviderAccess(
                    owner=owner.user_id, sharing=ProviderSharing.PRIVATE
                )
            },
        )
        with patch.object(mass_minimal, "metadata", MagicMock()):
            stored = await mass_minimal.music.tracks.add_item_to_library(
                create_track("spotify_theirs", "t1")
            )
        metadata_controller._update_track_metadata = AsyncMock()  # type: ignore[method-assign]
        crafted = create_track("spotify_theirs", "t1", name="Crafted")
        crafted.item_id, crafted.provider = stored.item_id, "library"

        for user in (member, owner):
            user_token = current_user.set(user)
            try:
                if user is member:
                    # the only source of the track is hidden from the member
                    with pytest.raises(MediaNotFoundError):
                        await metadata_controller.get_track_lyrics(crafted)
                else:
                    assert await metadata_controller.get_track_lyrics(crafted) == (None, None)
                assert get_current_user() is user
            finally:
                current_user.reset(user_token)
        metadata_controller._update_track_metadata.assert_not_awaited()
        assert get_current_user() is None
    finally:
        await mass_minimal.music.database.close()


def _lyrics_metadata_provider(metadata: MediaItemMetadata | None) -> MagicMock:
    """Return a lyrics metadata provider stub."""
    provider = MagicMock()
    provider.name = "LRCLIB"
    provider.priority = 0
    provider.supported_features = {ProviderFeature.LYRICS}
    provider.get_track_metadata = AsyncMock(return_value=metadata)
    return provider


def _music_lyrics_provider(
    instance_id: str, domain: str, *, streaming: bool = False, lyrics: bool = True
) -> MagicMock:
    """Return a visible music provider stub with optional lyrics support."""
    provider = MagicMock(spec=MusicProvider)
    provider.name = instance_id
    provider.instance_id = instance_id
    provider.domain = domain
    provider.is_streaming_provider = streaming
    provider.supported_features = {ProviderFeature.LYRICS} if lyrics else set()
    return provider


async def test_get_track_lyrics_prefers_synced_over_plain(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Synced metadata-provider lyrics win over the track's plain lyrics."""
    plain = "Plain source lyrics"
    synced = "[00:12.00]Synchronized lyrics"
    track = create_track("spotify_1", "track1")
    track.metadata.lyrics = plain
    provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics=synced))

    with patch.object(mass_minimal, "get_providers", return_value=[provider]):
        assert await metadata_controller.get_track_lyrics(track) == (plain, synced)

    provider.get_track_metadata.assert_awaited_once_with(track)


@pytest.mark.parametrize("lookup", [None, RuntimeError("provider unavailable")])
async def test_get_track_lyrics_falls_back_to_plain_when_synced_unavailable(
    metadata_controller: MetaDataController,
    mass_minimal: MusicAssistant,
    lookup: MediaItemMetadata | Exception | None,
) -> None:
    """Missing or failing secondary lyrics providers preserve the source text."""
    plain = "Plain source lyrics"
    track = create_track("spotify_1", "track1")
    track.metadata.lyrics = plain
    provider = _lyrics_metadata_provider(None)
    if isinstance(lookup, Exception):
        provider.get_track_metadata = AsyncMock(side_effect=lookup)
    else:
        provider.get_track_metadata = AsyncMock(return_value=lookup)

    with patch.object(mass_minimal, "get_providers", return_value=[provider]):
        assert await metadata_controller.get_track_lyrics(track) == (plain, None)

    provider.get_track_metadata.assert_awaited_once_with(track)


async def test_get_track_lyrics_respects_prefer_synced_disabled(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Disabling synced preference returns the first available lyrics without fallback."""
    track = create_track("spotify_1", "track1")
    track.metadata.lyrics = "Plain source lyrics"
    provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics="[00:12.00]Synced"))

    with (
        patch.object(metadata_controller.config, "get_value", return_value=False),
        patch.object(mass_minimal, "get_providers", return_value=[provider]) as get_providers,
    ):
        assert await metadata_controller.get_track_lyrics(track) == ("Plain source lyrics", None)

    get_providers.assert_not_called()
    provider.get_track_metadata.assert_not_awaited()


async def test_get_track_lyrics_returns_primary_synced_lyrics_immediately(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Timed lyrics already on the track take precedence over other providers."""
    synced = "[00:12.00]Synchronized lyrics"
    track = create_track("spotify_1", "track1")
    track.metadata.lyrics = synced
    provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics="[00:20.00]Other"))

    with patch.object(mass_minimal, "get_providers", return_value=[provider]) as get_providers:
        assert await metadata_controller.get_track_lyrics(track) == (synced, synced)

    get_providers.assert_not_called()
    provider.get_track_metadata.assert_not_awaited()


async def test_get_track_lyrics_uses_synced_provider_text_and_stops_lookup(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """The synced result's plain text wins, and later providers are not queried."""
    track = create_track("spotify_1", "track1")
    track.metadata.lyrics = "Track plain lyrics"
    first = _lyrics_metadata_provider(MediaItemMetadata(lyrics="Earlier provider lyrics"))
    synced = "[00:12.00]Synchronized lyrics"
    second = _lyrics_metadata_provider(
        MediaItemMetadata(lyrics="Synced provider plain lyrics", lrc_lyrics=synced)
    )
    last = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics="[00:20.00]Other"))
    first.priority, second.priority, last.priority = 0, 1, 2

    with patch.object(mass_minimal, "get_providers", return_value=[first, second, last]):
        assert await metadata_controller.get_track_lyrics(track) == (
            "Synced provider plain lyrics",
            synced,
        )

    first.get_track_metadata.assert_awaited_once_with(track)
    second.get_track_metadata.assert_awaited_once_with(track)
    last.get_track_metadata.assert_not_awaited()


async def test_get_track_lyrics_skips_providers_without_lyrics_feature(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Only metadata providers advertising lyrics are queried."""
    track = create_track("spotify_1", "track1")
    track.metadata.lyrics = "Plain source lyrics"
    unsupported = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics="[00:20.00]Other"))
    unsupported.supported_features = set()
    supported = _lyrics_metadata_provider(None)

    with patch.object(mass_minimal, "get_providers", return_value=[unsupported, supported]):
        assert await metadata_controller.get_track_lyrics(track) == ("Plain source lyrics", None)

    unsupported.get_track_metadata.assert_not_awaited()
    supported.get_track_metadata.assert_awaited_once_with(track)


async def test_get_track_lyrics_uses_resolved_provider_instance_for_primary_lyrics(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Plain lyrics use the resolved provider instance and survive an empty metadata lookup."""
    track = create_track("spotify_1", "track1")
    full_track = create_track("spotify_2", "track1")
    full_track.metadata.lyrics = "Primary provider lyrics"
    music_provider = MagicMock(spec=MusicProvider)
    music_provider.instance_id = "spotify_2"
    music_provider.supported_features = {ProviderFeature.LYRICS}
    metadata_provider = _lyrics_metadata_provider(None)
    mass_minimal.music = MusicController(mass_minimal)

    with (
        patch.object(mass_minimal.music, "get_visible_provider", return_value=music_provider),
        patch.object(
            mass_minimal.music.tracks, "get_provider_item", new=AsyncMock(return_value=full_track)
        ) as get_provider_item,
        patch.object(mass_minimal, "get_providers", return_value=[metadata_provider]),
    ):
        assert await metadata_controller.get_track_lyrics(track) == (
            "Primary provider lyrics",
            None,
        )

    get_provider_item.assert_awaited_once_with(
        track.item_id, music_provider.instance_id, strict_provider_instance=True
    )
    metadata_provider.get_track_metadata.assert_awaited_once_with(track)


async def test_get_track_lyrics_uses_stored_library_track_without_refresh(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Metadata providers receive the canonical item and only fetched lyrics are stored."""
    stored = create_track("spotify_1", "track1")
    stored.item_id, stored.provider = "1", "library"
    stored.metadata.lyrics = "Stored plain lyrics"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    metadata_controller._update_track_metadata = AsyncMock()  # type: ignore[method-assign]
    synced = "[00:12.00]Synchronized lyrics"
    provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics=synced))
    track = create_track("spotify_1", "track1")
    track.item_id, track.provider = "1", "library"

    with patch.object(mass_minimal, "get_providers", return_value=[provider]):
        assert await metadata_controller.get_track_lyrics(track) == ("Stored plain lyrics", synced)

    mass_minimal.music.tracks.get.assert_awaited_with("1", "library", allow_update_metadata=False)
    assert mass_minimal.music.tracks.get.await_count == 2
    mass_minimal.music.tracks.replace_metadata.assert_awaited_once_with(
        "1", {"lyrics": "Stored plain lyrics", "lrc_lyrics": synced}
    )
    metadata_controller._update_track_metadata.assert_not_awaited()
    provider.get_track_metadata.assert_awaited_once_with(stored)


async def test_empty_library_lookup_continues_to_metadata_providers(
    metadata_controller: MetaDataController,
    mass_minimal: MusicAssistant,
) -> None:
    """An empty library item continues to lyrics providers instead of refreshing metadata."""
    stored = create_track("spotify_1", "track1")
    stored.item_id, stored.provider = "1", "library"
    stored.metadata.genres = {"Stored genre"}
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    metadata_controller._update_track_metadata = AsyncMock()  # type: ignore[method-assign]
    synced = "[00:12.00]Synchronized lyrics"
    provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics=synced))
    track = create_track("spotify_1", "track1")
    track.item_id, track.provider = "1", "library"

    with (
        patch.object(metadata_controller.config, "get_value", return_value=False),
        patch.object(mass_minimal, "get_providers", return_value=[provider]),
    ):
        assert await metadata_controller.get_track_lyrics(track) == (None, synced)

    mass_minimal.music.tracks.get.assert_awaited_with("1", "library", allow_update_metadata=False)
    assert mass_minimal.music.tracks.get.await_count == 2
    metadata_controller._update_track_metadata.assert_not_awaited()
    provider.get_track_metadata.assert_awaited_once_with(stored)
    assert stored.metadata.genres == {"Stored genre"}
    assert stored.metadata.lyrics is None
    assert stored.metadata.lrc_lyrics is None
    mass_minimal.music.tracks.replace_metadata.assert_awaited_once_with(
        "1", {"lyrics": None, "lrc_lyrics": synced}
    )


async def test_get_track_lyrics_saves_only_normalized_lyrics_as_server(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """The caller's visible library item is validated before only lyrics are saved as server."""
    user = User(user_id="user-member", username="member", role=UserRole.USER)
    stored = create_track("spotify_1", "track1")
    stored.item_id, stored.provider = "1", "library"
    stored.provider_mappings = set()
    stored.metadata.genres = {"Stored genre"}
    track = create_track("spotify_1", "track1")
    track.item_id, track.provider = "1", "library"
    track.provider_mappings = set()
    provider_metadata = MediaItemMetadata(
        lyrics="Fetched plain lyrics",
        lrc_lyrics="[00:20.00]Later\n[00:10.00]Earlier",
        description="Provider description",
        genres={"Provider genre"},
    )
    provider = _lyrics_metadata_provider(provider_metadata)
    users_during_reads: list[User | None] = []
    users_during_writes: list[User | None] = []

    async def get_library_track(*_: object, **__: object) -> Track:
        users_during_reads.append(get_current_user())
        return stored

    async def replace_metadata(*_: object, **__: object) -> None:
        users_during_writes.append(get_current_user())

    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(side_effect=get_library_track)
    mass_minimal.music.tracks.replace_metadata = AsyncMock(side_effect=replace_metadata)

    user_token = current_user.set(user)
    try:
        with patch.object(mass_minimal, "get_providers", return_value=[provider]):
            assert await metadata_controller.get_track_lyrics(track) == (
                "Fetched plain lyrics",
                "[00:10.00]Earlier\n[00:20.00]Later",
            )
        assert get_current_user() is user
    finally:
        current_user.reset(user_token)

    assert users_during_reads == [user, user]
    assert users_during_writes == [None]
    mass_minimal.music.tracks.replace_metadata.assert_awaited_once_with(
        "1",
        {
            "lyrics": "Fetched plain lyrics",
            "lrc_lyrics": "[00:10.00]Earlier\n[00:20.00]Later",
        },
    )
    assert stored.metadata.genres == {"Stored genre"}


async def test_get_track_lyrics_does_not_clear_stored_lyrics(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """An empty provider result does not clear existing lyrics from a library item."""
    stored = create_track("spotify_1", "track1")
    stored.item_id, stored.provider = "1", "library"
    stored.metadata.lyrics = "Stored plain lyrics"
    track = create_track("spotify_1", "track1")
    track.item_id, track.provider = "1", "library"
    track.metadata.lyrics = "Stored plain lyrics"
    provider = _lyrics_metadata_provider(None)
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()

    with patch.object(mass_minimal, "get_providers", return_value=[provider]):
        assert await metadata_controller.get_track_lyrics(track) == ("Stored plain lyrics", None)

    mass_minimal.music.tracks.replace_metadata.assert_not_awaited()


async def test_cached_library_synced_lyrics_check_stored_item_for_local_sources(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Cached lyrics still require checking the canonical library mappings for overrides."""
    track = create_track("spotify_1", "track1")
    track.item_id, track.provider = "1", "library"
    track.metadata.lrc_lyrics = "[00:12.00]Cached library lyrics"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=track)
    metadata_controller._update_track_metadata = AsyncMock()  # type: ignore[method-assign]

    with patch.object(mass_minimal, "get_providers", return_value=[]):
        assert await metadata_controller.get_track_lyrics(track) == (
            None,
            "[00:12.00]Cached library lyrics",
        )

    mass_minimal.music.tracks.get.assert_awaited_once_with(
        "1", "library", allow_update_metadata=False
    )
    metadata_controller._update_track_metadata.assert_not_awaited()


async def test_stored_library_synced_lyrics_stop_before_provider_discovery(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Synchronized lyrics on the stored item stop lookup before any provider discovery."""
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    synced = "[00:12.00]Stored synchronized lyrics"
    stored.metadata.lrc_lyrics = synced
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()

    with patch.object(mass_minimal, "get_providers") as get_providers:
        assert await metadata_controller.get_track_lyrics(track) == (None, synced)

    mass_minimal.music.get_visible_provider.assert_not_called()
    get_providers.assert_not_called()
    mass_minimal.music.tracks.replace_metadata.assert_awaited_once_with(
        "1", {"lyrics": None, "lrc_lyrics": synced}
    )


@pytest.mark.parametrize("prefer_synced", [True, False])
async def test_local_plain_lyrics_override_cached_synced_lyrics(
    metadata_controller: MetaDataController,
    mass_minimal: MusicAssistant,
    prefer_synced: bool,
) -> None:
    """Fresh plain lyrics from a local file replace an older synchronized result."""
    mapping = ProviderMapping(
        item_id="local-track",
        provider_domain="filesystem_local",
        provider_instance="filesystem_local_1",
    )
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    stored.provider_mappings = {mapping}
    stored.metadata.lrc_lyrics = "[00:02.00]Cached online lyrics"
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    track.provider_mappings = set()
    track.metadata.lrc_lyrics = "[00:02.00]Cached online lyrics"
    local_provider = _music_lyrics_provider("filesystem_local_1", "filesystem_local")
    local_track = create_track("filesystem_local_1", "local-track")
    local_track.metadata.lyrics = "Fresh local plain lyrics"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.get_provider_item = AsyncMock(return_value=local_track)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    mass_minimal.music.get_visible_provider.return_value = local_provider
    metadata_controller._update_track_metadata = AsyncMock()  # type: ignore[method-assign]

    with (
        patch.object(metadata_controller.config, "get_value", return_value=prefer_synced),
        patch.object(mass_minimal, "get_providers", return_value=[]),
    ):
        assert await metadata_controller.get_track_lyrics(track) == (
            "Fresh local plain lyrics",
            None,
        )

    mass_minimal.music.tracks.get_provider_item.assert_awaited_once_with(
        "local-track",
        "filesystem_local_1",
        force_refresh=True,
        allow_fallback=False,
        strict_provider_instance=True,
    )
    mass_minimal.music.tracks.replace_metadata.assert_awaited_once_with(
        "1", {"lyrics": "Fresh local plain lyrics", "lrc_lyrics": None}
    )


@pytest.mark.parametrize("provider_visible", [True, False])
async def test_unavailable_local_provider_does_not_replace_stored_lyrics(
    metadata_controller: MetaDataController,
    mass_minimal: MusicAssistant,
    provider_visible: bool,
) -> None:
    """An inaccessible local source cannot clear the lyrics already stored in the library."""
    mapping = ProviderMapping(
        item_id="local-track",
        provider_domain="filesystem_local",
        provider_instance="filesystem_local_1",
    )
    synced = "[00:02.00]Stored lyrics"
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    stored.provider_mappings = {mapping}
    stored.metadata.lrc_lyrics = synced
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    track.provider_mappings = {mapping}
    track.metadata.lrc_lyrics = synced
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.get_provider_item = AsyncMock(
        side_effect=MediaNotFoundError("Local file unavailable")
    )
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    mass_minimal.music.get_visible_provider.return_value = (
        _music_lyrics_provider("filesystem_local_1", "filesystem_local")
        if provider_visible
        else None
    )

    assert await metadata_controller.get_track_lyrics(track) == (None, synced)

    assert mass_minimal.music.tracks.get_provider_item.await_count == int(provider_visible)
    mass_minimal.music.tracks.replace_metadata.assert_not_awaited()


async def test_direct_local_plain_lyrics_stop_online_lookup(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Plain local overrides also apply to tracks that are not in the library."""
    track = create_track("filesystem_local_1", "local-track")
    track.provider_mappings = {
        ProviderMapping(
            item_id="local-track",
            provider_domain="filesystem_local",
            provider_instance="filesystem_local_1",
        )
    }
    track.metadata.lyrics = "Local plain lyrics"
    online_provider = _lyrics_metadata_provider(
        MediaItemMetadata(lrc_lyrics="[00:02.00]Online synchronized lyrics")
    )
    mass_minimal.music = MagicMock()
    mass_minimal.music.get_visible_provider.return_value = _music_lyrics_provider(
        "filesystem_local_1", "filesystem_local"
    )
    mass_minimal.music.tracks.get_provider_item = AsyncMock(return_value=track)

    with patch.object(mass_minimal, "get_providers", return_value=[online_provider]):
        assert await metadata_controller.get_track_lyrics(track) == ("Local plain lyrics", None)

    online_provider.get_track_metadata.assert_not_awaited()


async def test_library_music_sources_follow_mapping_priority_and_stop_on_synced_lyrics(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Mapped music sources are queried by priority before metadata providers."""
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    local_mapping = next(iter(create_track("local_1", "local-track").provider_mappings))
    local_mapping.in_library = True
    streaming_mapping = next(iter(create_track("spotify_1", "stream-track").provider_mappings))
    streaming_mapping.in_library = False
    stored.provider_mappings = {local_mapping, streaming_mapping}
    local_provider = _music_lyrics_provider("local_1", "local", streaming=False)
    streaming_provider = _music_lyrics_provider("spotify_1", "spotify", streaming=True)
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    mass_minimal.music.tracks.get_provider_item = AsyncMock()
    metadata_controller._update_track_metadata = AsyncMock()  # type: ignore[method-assign]
    mass_minimal.music.get_visible_provider.side_effect = lambda provider, **_: {
        "local_1": local_provider,
        "spotify_1": streaming_provider,
    }.get(provider)
    metadata_provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics="[00:12.00]Synced"))
    synced = "[00:12.00]Synced"
    second_track = create_track("spotify_1", "stream-track")
    second_track.metadata.lrc_lyrics = synced
    mass_minimal.music.tracks.get_provider_item.side_effect = [
        create_track("local_1", "local-track"),
        second_track,
    ]

    with patch.object(mass_minimal, "get_providers", return_value=[metadata_provider]):
        assert await metadata_controller.get_track_lyrics(track) == (None, synced)

    assert [
        call.args[:2] for call in mass_minimal.music.tracks.get_provider_item.await_args_list
    ] == [
        ("local-track", "local_1"),
        ("stream-track", "spotify_1"),
    ]
    assert all(
        call.kwargs["strict_provider_instance"]
        for call in mass_minimal.music.tracks.get_provider_item.await_args_list
    )
    metadata_provider.get_track_metadata.assert_not_awaited()
    metadata_controller._update_track_metadata.assert_not_awaited()


async def test_duplicate_streaming_provider_mappings_are_queried_once(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Mappings from multiple instances of one streaming provider are deduplicated."""
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    mapping_one = next(iter(create_track("spotify_1", "provider-track-1").provider_mappings))
    mapping_two = next(iter(create_track("spotify_2", "provider-track-2").provider_mappings))
    stored.provider_mappings = {mapping_one, mapping_two}
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    first_provider = _music_lyrics_provider("spotify_1", "spotify", streaming=True)
    second_provider = _music_lyrics_provider("spotify_2", "spotify", streaming=True)
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    providers = {"spotify_1": first_provider, "spotify_2": second_provider}
    mass_minimal.music.get_visible_provider.side_effect = lambda provider, **_: providers.get(
        provider
    )
    fetched_tracks: list[Track] = []

    async def get_provider_item(item_id: str, *_: object, **__: object) -> Track:
        result = create_track("spotify_1", item_id)
        if fetched_tracks:
            result.metadata.lrc_lyrics = "[00:12.00]Second instance lyrics"
        fetched_tracks.append(result)
        return result

    mass_minimal.music.tracks.get_provider_item = AsyncMock(side_effect=get_provider_item)
    metadata_provider = _lyrics_metadata_provider(
        MediaItemMetadata(lrc_lyrics="[00:15.00]Metadata provider lyrics")
    )

    with patch.object(mass_minimal, "get_providers", return_value=[metadata_provider]):
        assert await metadata_controller.get_track_lyrics(track) == (
            None,
            "[00:15.00]Metadata provider lyrics",
        )

    mass_minimal.music.tracks.get_provider_item.assert_awaited_once()
    metadata_provider.get_track_metadata.assert_awaited_once_with(stored)


async def test_library_music_sources_skip_hidden_unsupported_and_unavailable_mappings(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """Only available, visible mappings with lyrics support are queried."""
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    unavailable_mapping = next(
        iter(create_track("unavailable_1", "unavailable-track").provider_mappings)
    )
    unavailable_mapping.available = False
    unsupported_mapping = next(
        iter(create_track("unsupported_1", "unsupported-track").provider_mappings)
    )
    hidden_mapping = next(iter(create_track("hidden_1", "hidden-track").provider_mappings))
    stored.provider_mappings = {unavailable_mapping, unsupported_mapping, hidden_mapping}
    unsupported_provider = _music_lyrics_provider("unsupported_1", "unsupported", lyrics=False)
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    mass_minimal.music.tracks.get_provider_item = AsyncMock()
    mass_minimal.music.get_visible_provider.side_effect = lambda provider, **_: {
        "unsupported_1": unsupported_provider,
        "hidden_1": None,
    }.get(provider)
    synced = "[00:12.00]Metadata provider lyrics"
    metadata_provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics=synced))

    with patch.object(mass_minimal, "get_providers", return_value=[metadata_provider]):
        assert await metadata_controller.get_track_lyrics(track) == (None, synced)

    mass_minimal.music.tracks.get_provider_item.assert_not_awaited()
    metadata_provider.get_track_metadata.assert_awaited_once_with(stored)
    assert mass_minimal.music.get_visible_provider.call_count == 2
    assert {call.args[0] for call in mass_minimal.music.get_visible_provider.call_args_list} == {
        "unsupported_1",
        "hidden_1",
    }


async def test_library_mapped_source_failure_continues_to_metadata_provider(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """A failed mapped music source does not prevent metadata-provider lyrics lookup."""
    stored = create_track("spotify_1", "library-track")
    stored.item_id, stored.provider = "1", "library"
    mapping = next(iter(create_track("spotify_1", "provider-track").provider_mappings))
    stored.provider_mappings = {mapping}
    music_provider = _music_lyrics_provider("spotify_1", "spotify", streaming=True)
    track = create_track("spotify_1", "library-track")
    track.item_id, track.provider = "1", "library"
    mass_minimal.music = MagicMock()
    mass_minimal.music.tracks.get = AsyncMock(return_value=stored)
    mass_minimal.music.tracks.replace_metadata = AsyncMock()
    mass_minimal.music.tracks.get_provider_item = AsyncMock(
        side_effect=RuntimeError("music source unavailable")
    )
    mass_minimal.music.get_visible_provider.return_value = music_provider
    synced = "[00:12.00]Metadata provider lyrics"
    metadata_provider = _lyrics_metadata_provider(MediaItemMetadata(lrc_lyrics=synced))

    with patch.object(mass_minimal, "get_providers", return_value=[metadata_provider]):
        assert await metadata_controller.get_track_lyrics(track) == (None, synced)

    mass_minimal.music.tracks.get_provider_item.assert_awaited_once_with(
        "provider-track", "spotify_1", strict_provider_instance=True
    )
    metadata_provider.get_track_metadata.assert_awaited_once_with(stored)


async def test_prefer_synced_lyrics_config_defaults_to_enabled(
    metadata_controller: MetaDataController,
) -> None:
    """The Metadata controller exposes the new preference enabled by default."""
    entries = await metadata_controller.get_config_entries()
    entry = next(entry for entry in entries if entry.key == CONF_PREFER_SYNCED_LYRICS)
    assert entry.default_value is True
