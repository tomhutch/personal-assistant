import logging

import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

import config

logger = logging.getLogger(__name__)


class SpotifySearchError(Exception):
    """A Spotify search failure with a message suitable for voice feedback."""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def find_spotify_media(query: str) -> tuple[str, str, str]:
    """Return (spotify_uri, media_type, display_name) for a spoken search phrase.

    Searching the public catalog only needs the app's client credentials, so no
    user login or token cache is involved; playback is handled by Home Assistant.
    """
    if not config.SPOTIPY_CLIENT_ID or not config.SPOTIPY_CLIENT_SECRET:
        raise SpotifySearchError(
            "Spotify search isn't configured. Add SPOTIPY_CLIENT_ID and "
            "SPOTIPY_CLIENT_SECRET to the .env file."
        )

    try:
        client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=config.SPOTIPY_CLIENT_ID,
                client_secret=config.SPOTIPY_CLIENT_SECRET,
                cache_handler=spotipy.MemoryCacheHandler(),
            ),
            requests_timeout=10,
        )
        results = client.search(q=query, type="artist,track,album,playlist", limit=1)
    except (SpotifyException, spotipy.SpotifyOauthError) as error:
        logger.error("Spotify search failed: %s", error)
        raise SpotifySearchError("I couldn't search Spotify right now.") from error
    except Exception as error:
        logger.error("Couldn't reach Spotify: %s", error)
        raise SpotifySearchError("I couldn't connect to Spotify.") from error

    def first(kind: str) -> dict | None:
        items = (results.get(f"{kind}s") or {}).get("items") or []
        return next((item for item in items if item), None)

    artist, track, album, playlist = (
        first(kind) for kind in ("artist", "track", "album", "playlist")
    )
    # An exact artist-name match means "play <artist>"; otherwise prefer a song.
    if artist and _normalize(artist.get("name", "")) == _normalize(query):
        chosen, media_type = artist, "artist"
    else:
        chosen, media_type = next(
            (
                (item, kind)
                for item, kind in (
                    (track, "track"),
                    (album, "album"),
                    (playlist, "playlist"),
                    (artist, "artist"),
                )
                if item
            ),
            (None, ""),
        )
    if not chosen or not chosen.get("uri"):
        raise SpotifySearchError(f"I couldn't find {query} on Spotify.")
    if media_type == "artist":
        # Sonos can't play artist URIs, so play the artist's top-ranked album instead.
        name = chosen.get("name") or query
        try:
            albums = (
                client.search(q=f'artist:"{name}"', type="album", limit=1).get("albums") or {}
            ).get("items") or []
        except Exception as error:
            logger.error("Spotify artist album search failed: %s", error)
            albums = []
        album = next((item for item in albums if item and item.get("uri")), None)
        if not album:
            raise SpotifySearchError(f"I couldn't find anything to play for {name}.")
        return album["uri"], "album", f"{album.get('name') or name} by {name}"
    return chosen["uri"], media_type, chosen.get("name") or query
