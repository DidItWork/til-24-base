#!/bin/bash

docker build -t backstreet-boys-autonomy .

docker tag backstreet-boys-autonomy asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-autonomy:finals

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-autonomy:finals