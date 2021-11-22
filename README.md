# Nebo

> Nebo is a noun meaning the Babylonian god of wisdom and agriculture and patron of scribes and schools

### Introduction

This project aims to combine computer vision and natural language processing to build a tool that can be used by anyone (from agricultural researchers, farmers, and hobbyists).

Using object detection, the prediction and localization of plant disease can be easily detected, and further combining this with question answering model feeding questions about the pathogen, the user gets to learn about what's ailing the crops, how to treat them, and a whole range of other agricultural-related questions.

### Training

The dataset currently used to train the model is custom-built (a small sample, I couldn't find publicly available annotated datasets for this particular use case).

The model performance is expected to improve as more data is added and a newer version of the model is deployed.

> ## Outdated instructions, updating the README soon.
>
> #### but if you know how to use kubectl and customize, clone the repo and apply the manifests in the deploy folder.

### Dependencies

To set up the environment to run the code in this repository, follow the instructions below.

1. Install docker (avoid the "it works on my machine 😅")

```bash
# Download Docker
curl -fsSL get.docker.com -o get-docker.sh
# Install Docker using the stable channel (instead of the default "edge")
CHANNEL=stable sh get-docker.sh
# Remove Docker install script
rm get-docker.sh
```

2. Install docker compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

```

3. Clone the repository (if you haven't already!), and navigate to the `box_of_crayons` folder, then use docker-compose to start the applications

```bash
git clone https://github.com/mrdvince/box_of_crayons.gitt
cd box_of_crayons
```

> NB: You might have to set a couple of env variables

4. Start the application.

```bash
docker-compose up -d
```

this will pull and build all the images required to run the containers and finally start it.

> Going to switch to Kubernetes at some point but right now docker swarm seems alright
