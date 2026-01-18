username=yyang
project_name=sam3d
container_name=sam3d
folder_name=sam3d

docker run --gpus all -itd \
    -u $(id -u $username):$(id -g $username) \
    --name ${username}_${container_name} \
    -v /mnt/workspace2024/${username}/${folder_name}:/home/${username}/mnt/workspace \
    --mount type=bind,source="/mnt/poplin/share/2023/users/yang/",target=/mnt/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    repo-luna.ist.osaka-u.ac.jp:5000/yyang/sam3d:build \
