# DRL Homework 2

### Topics
- DQN
- Rainbow
    - Double Q-learning
    - Prioritized Replay
    - Dueling Network
    - Multi-step Learning
    - Distributional RL
    - Noisy Nets

### Structure

Follow the "hw.ipynb" ipython notebook for instructions:


You can run the test script that covers replay buffer and related implementations from the "test" directory using:

```
python -m unittest test_replaybuffer.py
```

### Installation

To start your homework, you need to install this package and requirements. You can install requirements with the following command in the homework directory:

```
pip install -r requirements.txt
```
Then you need to install the homework package. You can install the package with the following command:

```
pip install -e .
```

This command will install the homework package in development mode so that the installation location will be the current directory.

### Docker

You can also use docker to work on your homework. Simply build a docker image from the homework directory using the following command:

```
docker build -t rlhw2 .
```

You may need to install docker first if you don't have it already.

After building a container we need to mount the homework directory at your local computer to the container we want to run. Note that the container will install necessary python packages in build.

You can run the container using the command below as long as your current directory is the homework directory:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw2 rlhw2
```

This way you can connect the container at ```localhost:8889``` in your browser. Note that, although we are using docker, changes are made in your local directory since we mounted it.

You can also use it interactively by simply running:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw2 rlhw2 /bin/bash
```

> Note: Running docker with cuda requires additional steps!

### Submission

You need to submit this repository after you filled it (and additional files that you used if there happens to be any). You also need to fill "logs" directory with the results of your experiments as instructed in the ipython notebook. Submissions are done via Ninova until the submission deadline. For the model parameters, you should put a google drive link to the ipython notebook.

### Evaluation

- Experiments 40%
- Implementations 70%

The total score will be clipped to 100%.

- DQN (25%)
- RAINBOW (75%)
    - Prioritized Replay 20%
    - Distributional RL 20%
    - Noisy Nets 15%
    - Multi-step learning 10%
    - Dueling Networks 5%
    - Double Q-learning 5%


### Related Readings

- [DQN](https://www.nature.com/articles/nature14236)
- [Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [Dueling Network](https://arxiv.org/pdf/1511.06581.pdf)
- Multi-step Learning - Richard S. Sutton and Andrew G. Barto Chapter 7
- [Distributional RL](https://arxiv.org/pdf/1707.06887.pdf)
- [Noisy Nets](https://arxiv.org/pdf/1706.10295.pdf)
- [Rainbow](https://arxiv.org/pdf/1710.02298.pdf)

### Questions

You can ask any question related to the homework via:

okt@itu.edu.tr
