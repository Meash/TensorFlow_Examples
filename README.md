# TensorFlow Examples

Sample use cases for TensorFlow, demonstrating its applications.


## Run in Docker

Run Docker Container, opening port 8888

    docker run -it --name tensorflow -p 8888:8888 b.gcr.io/tensorflow/tensorflow
Run IPython Notebook

    ipython notebook &

On the host machine, find out the host ip
    
    HOST_IP=$(docker-machine ip default)
    IPYTHON_NOTEBOOK_ADDR="http://${HOST_IP}:8888/"
    echo $IPYTHON_NOTEBOOK_ADDR

Connect to that address from your IDE or open it in your browser.


## Other demos
Also check out https://github.com/aymericdamien/TensorFlow-Examples/, many interesting demos there as well.
