#!/usr/bin/env bash

set -e  # stop on first error

ROOT_DIR=$(dirname $(dirname $(realpath "$0")))
echo "ROOT_DIR=$ROOT_DIR"


if [ -f "$ROOT_DIR/.local/settings.env" ]; then
  source $ROOT_DIR/.local/settings.env
fi

# you can override in your .local/secretes.env
COMPOSE_FILES=${COMPOSE_FILES:=-f compose.yaml}
echo "COMPOSE_FILES=$COMPOSE_FILES"


docker compose $COMPOSE_FILES -p jnbooks down
# docker container prune -f
