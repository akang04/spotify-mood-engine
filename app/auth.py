import os

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

SCOPES = " ".join([
    "user-library-read",
    "user-top-read",
    "user-read-recently-played",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-read-private",
])

_CACHE_PATH = ".spotify_cache"


def get_oauth_manager() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ["SPOTIPY_REDIRECT_URI"],
        scope=SCOPES,
        cache_path=_CACHE_PATH,
        show_dialog=False,
    )


def get_spotify_client() -> spotipy.Spotify:
    """Return an authenticated Spotipy client, refreshing the token if needed."""
    manager = get_oauth_manager()
    token_info = manager.get_cached_token()

    if not token_info:
        raise RuntimeError("No cached token found. Complete the OAuth flow first.")

    if manager.is_token_expired(token_info):
        token_info = manager.refresh_access_token(token_info["refresh_token"])

    return spotipy.Spotify(
        auth=token_info["access_token"],
        retries=3,
        status_forcelist={429, 500, 502, 503, 504},
        backoff_factor=1,
    )


def get_auth_url() -> str:
    """Return the Spotify authorization URL to redirect the user to."""
    return get_oauth_manager().get_authorize_url()


def exchange_code(code: str) -> dict:
    """Exchange an authorization code for token info and cache it."""
    manager = get_oauth_manager()
    token_info = manager.get_access_token(code, as_dict=True, check_cache=False)
    return token_info


def get_current_user(sp: spotipy.Spotify) -> dict:
    """Return the current user's Spotify profile."""
    return sp.current_user()
