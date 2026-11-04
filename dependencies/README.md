# Dependencies

Optional services the assistant works alongside. They are deployed separately
from `setup.sh` because they may run on a different machine (for example, the
one running Home Assistant) and keep their own persistent data.

## Music Assistant

Runs the [Music Assistant](https://music-assistant.io) server in Docker so
Home Assistant can play Spotify artists and playlists on your speakers.

```bash
./dependencies/music-assistant/deploy.sh
```

Requirements: Docker with the Compose plugin, and a host on the same network as
your speakers (the container uses host networking). The script is idempotent,
so re-running it pulls the latest image and restarts the container if needed.

- `MA_DATA_DIR`: persistent data directory (default `~/docker/music-assistant-server/data`). Keep it between updates.
- `MA_LOG_LEVEL`: `critical`, `error`, `warning`, `info` (default) or `debug`.

Do not switch between the stable and beta image tags in place, and do not move
the database between stable and beta instances.

After it starts, open `http://<host>:8095`, add Spotify as a provider, then add
the Music Assistant integration in Home Assistant.
