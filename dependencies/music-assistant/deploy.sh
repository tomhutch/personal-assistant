#!/usr/bin/env bash
#
# Deploys the Music Assistant server (https://music-assistant.io) with Docker.
# Run this on a machine on the same network as your speakers, ideally the one
# running Home Assistant. It is deliberately separate from the repo's setup.sh.
#
# Optional environment variables:
#   MA_DATA_DIR   Persistent data directory (default: ~/docker/music-assistant-server/data)
#   MA_LOG_LEVEL  critical | error | warning | info | debug (default: info)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MA_DATA_DIR="${MA_DATA_DIR:-$HOME/docker/music-assistant-server/data}"
export MA_DATA_DIR

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Install it first: https://docs.docker.com/engine/install/" >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "The Docker Compose plugin is required (docker compose)." >&2
    exit 1
fi

mkdir -p "$MA_DATA_DIR"

docker compose -f "$SCRIPT_DIR/docker-compose.yml" pull
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "Music Assistant is starting. Open http://${HOST_IP:-<this-host>}:8095 to finish setup."
echo "Then add the Music Assistant integration in Home Assistant and Spotify as a provider."
