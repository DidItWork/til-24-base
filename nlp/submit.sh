#! /bin/bash

docker build -t backstreet-boys-nlp .

docker tag backstreet-boys-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-nlp:latest

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-nlp:latest

gcloud ai models upload --region asia-southeast1 --display-name 'backstreet-boys-nlp' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-nlp:latest --container-health-route /health --container-predict-route /extract --container-ports 5002 --version-aliases default