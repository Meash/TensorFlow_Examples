# TensorFlow Examples

Sample use cases for TensorFlow, demonstrating its applications.


## Run in Docker

Run Docker Container, forwarding port 8888 (IPython) and 6006 (TensorBoard)

    docker run -it --name tensorflow -p 8888:8888 -p 6006:6006 b.gcr.io/tensorflow/tensorflow

Run IPython Notebook

    ipython notebook &

On the host machine, find out the host ip (`default` being the name of the machine)
    
    echo "http://$(docker-machine ip default):8888"

Connect to that address from your IDE or open it in your browser.


## Run TensorBoard

    tensorboard /path/to/log-directory

You can connect to TensorBoard with your browser using the following ip:

    echo "http://$(docker-machine ip default):6006"


### Troubleshooting
If you cannot resolve hostnames on your docker container, try adding the dns ip manually by specifying `--dns=8.8.8.8` (using the Google DNS).
The run command then looks like this:

    docker run -it --dns=8.8.8.8 --name tensorflow_dns -p 8888:8888 b.gcr.io/tensorflow/tensorflow


## Other demos
Also check out https://github.com/aymericdamien/TensorFlow-Examples/, many interesting demos there as well.
