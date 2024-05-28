#! /bin/bash

docker build -t backstreet-boys-vlm .

docker tag backstreet-boys-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-vlm:latest

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-vlm:latest

gcloud ai models upload --region asia-southeast1 --display-name 'backstreet-boys-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-backstreet-boys/backstreet-boys-vlm:latest --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default