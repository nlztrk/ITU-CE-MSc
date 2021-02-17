# DRL Homework 3

### Topics
- Policy Gradient
  - REINFORCE
  - A2C

### Structure

Follow the "hw.ipynb" ipython notebook for instructions:

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
docker build -t rlhw3 .
```

You may need to install docker first if you don't have it already.

After building a container we need to mount the homework directory at your local computer to the container we want to run. Note that the container will install necessary python packages in build.

You can run the container using the command below as long as your current directory is the homework directory:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw3 rlhw3
```

This way you can connect the container at ```localhost:8889``` in your browser. Note that, although we are using docker, changes are made in your local directory since we mounted it.

You can also use it interactively by simply running:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw3 rlhw3 /bin/bash
```

> Note: Running docker with cuda requires additional steps!

### Submission

You need to submit this repository after you filled it (and additional files that you used if there happens to be any). You also need to fill "logs" and "models" directories with the results of your experiments as instructed in the ipython notebook. Submissions are done via Ninova until the submission deadline.

### Evaluation

- Experiments 50%
- Implementations 50%

The total score will be clipped to 100%.

- REINFORCE (25%)
- A2C (75%)

### Related Readings

- Reinforcement Learning: An Introduction (2nd), Richard S. Sutton and Andrew G. Barto Chapter 13
- [A3C](https://arxiv.org/abs/1602.01783)
- [Additional blog post for A2C](https://openai.com/blog/baselines-acktr-a2c/)

### Questions

You can ask any question related to the homework via:

okt@itu.edu.tr
