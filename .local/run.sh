#!/usr/bin/env bash

set -e  # stop on first error

ROOT_DIR=$(dirname $(dirname $(realpath "$0")))
echo "ROOT_DIR=$ROOT_DIR"

# get the IP address of the host machine for use in docker compose
# this fixes the issue of docker containers not able to reach services through bastion
export DOCKER_HOST_IP=$(ip addr show eth0 | grep "inet " | awk '{ print $2 }' | awk -F "/" '{ print $1 }' | head -1)
echo "DOCKER_HOST_IP: $DOCKER_HOST_IP"


if [ -f "$ROOT_DIR/.local/settings.env" ]; then
  source $ROOT_DIR/.local/settings.env
  DOCKER_ENV="--env-file \"$ROOT_DIR/.local/secrets.env\""
fi

# you can override in your .local/settings.env
COMPOSE_FILES=${COMPOSE_FILES:=-f compose.yaml}
echo "COMPOSE_FILES=$COMPOSE_FILES"


if [ "$1" = "--build" ]; then
  docker build -t my-jnbooks .
fi


docker compose $DOCKER_ENV $COMPOSE_FILES -p jnbooks up
