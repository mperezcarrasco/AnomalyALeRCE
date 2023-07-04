#!/usr/bin/env bash
docker build -t anomalyalerce \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .
