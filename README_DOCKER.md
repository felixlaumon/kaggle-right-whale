## Training models with Docker

Upload cache to S3

    aws s3 mb s3://kaggle-right-whale

    aws s3 cp cache/X_cropped_299_head_20151227.npy s3://kaggle-right-whale/cache/
    aws s3 cp cache/y_cropped_299_head_20151227.npy s3://kaggle-right-whale/cache/

    aws s3 cp cache/X_cropped_256_head_20151216.npy s3://kaggle-right-whale/cache/
    aws s3 cp cache/y_cropped_256_head_20151216.npy s3://kaggle-right-whale/cache/

    aws s3 cp models/encoder.pkl s3://kaggle-right-whale/models/

Download cache to host

    eval "$(docker-machine env aws01)"
    docker-machine ssh aws01

    sudo su -
    apt-get install awscli -y
    aws configure
    # Input the access key and secret and the region

    mkdir -p /mnt/kaggle-right-whale/cache
    mkdir -p /mnt/kaggle-right-whale/models
    aws s3 cp s3://kaggle-right-whale/models/encoder.pkl /mnt/kaggle-right-whale/models/encoder.pkl

    aws s3 cp s3://kaggle-right-whale/cache/X_cropped_299_head_20151227.npy /mnt/kaggle-right-whale/cache/
    aws s3 cp s3://kaggle-right-whale/cache/y_cropped_299_head_20151227.npy /mnt/kaggle-right-whale/cache/

    aws s3 cp s3://kaggle-right-whale/cache/X_cropped_256_head_20151216.npy /mnt/kaggle-right-whale/cache/
    aws s3 cp s3://kaggle-right-whale/cache/y_cropped_256_head_20151216.npy /mnt/kaggle-right-whale/cache/


Sync models back to S3 every 1 minute

    docker-machine ssh aws01
    sudo su -
    crontab -e
    * * * * * /usr/bin/aws s3 sync /mnt/kaggle-right-whale/models/ s3://kaggle-right-whale/models/

Build image

    docker build -t felixlaumon/kaggle-right-whale .

Train model

    docker run -ti \
        -d \
        --device /dev/nvidia0:/dev/nvidia0 \
        --device /dev/nvidiactl:/dev/nvidiactl \
        --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        -v /mnt/kaggle-right-whale/cache:/kaggle-right-whale/cache \
        -v /mnt/kaggle-right-whale/models:/kaggle-right-whale/models \
        felixlaumon/kaggle-right-whale \
        [command]

To view training logs

    docker logs -f [jobid]

Sanity check container

    docker run -ti \
        --device /dev/nvidia0:/dev/nvidia0 \
        --device /dev/nvidiactl:/dev/nvidiactl \
        --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        -v /mnt/kaggle-right-whale/cache:/kaggle-right-whale/cache \
        -v /mnt/kaggle-right-whale/models:/kaggle-right-whale/models \
        felixlaumon/kaggle-right-whale \
        /bin/bash

To start over

    docker stop $(docker ps -a -q)
    docker rm $(docker ps -a -q)
    docker rmi felixlaumon/kaggle-right-whale
