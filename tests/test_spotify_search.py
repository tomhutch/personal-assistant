from unittest.mock import MagicMock

import pytest

import spotify_search
from spotify_search import SpotifySearchError, find_spotify_media


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(spotify_search.config, "SPOTIPY_CLIENT_ID", "id")
    monkeypatch.setattr(spotify_search.config, "SPOTIPY_CLIENT_SECRET", "secret")
    monkeypatch.setattr(spotify_search.spotipy, "Spotify", lambda **kw: mock)
    mock = MagicMock()
    return mock


def _results(**kinds):
    return {
        f"{kind}s": {"items": [{"uri": f"spotify:{kind}:1", "name": name}]}
        for kind, name in kinds.items()
    }


def test_exact_artist_match_plays_top_album(client):
    client.search.side_effect = [
        _results(artist="Taylor Swift", track="Taylor Swift Song"),
        _results(album="1989"),
    ]
    assert find_spotify_media("taylor swift") == (
        "spotify:album:1", "album", "1989 by Taylor Swift"
    )
    assert client.search.call_args.kwargs["q"] == 'artist:"Taylor Swift"'


def test_prefers_track_over_other_results(client):
    client.search.return_value = _results(artist="AC/DC", track="Thunderstruck", album="The Razors Edge")
    assert find_spotify_media("thunderstruck") == ("spotify:track:1", "track", "Thunderstruck")


def test_no_results_raises(client):
    client.search.return_value = {"tracks": {"items": []}}
    with pytest.raises(SpotifySearchError):
        find_spotify_media("zzzz")


def test_missing_credentials_raises(monkeypatch):
    monkeypatch.setattr(spotify_search.config, "SPOTIPY_CLIENT_ID", "")
    with pytest.raises(SpotifySearchError):
        find_spotify_media("anything")
