#!/bin/bash

docker build -t backstreet-boys-main .

docker tag backstreet-boys-main asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-main:finals

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-main:finals