#!/bin/bash
for i in {1..5}; do
    cp -rn app/ "app$i/"
done

docker-compose up -d
