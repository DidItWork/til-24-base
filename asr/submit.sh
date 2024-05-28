#! /bin/bash

docker build -t backstreet-boys-asr .

docker tag backstreet-boys-asr asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-asr:latest

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-asr:latest

gcloud ai models upload --region asia-southeast1 --display-name 'backstreet-boys-asr' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-asr:latest --container-health-route /health --container-predict-route /stt --container-ports 5001 --version-aliases default
