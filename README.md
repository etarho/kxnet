# deepest3d

3D coordinate estimation of surgical instruments in laparoscopic surgery using deep neural network

## Prerequisites

- Docker Version: 19.03.8

## How to run

**1. To build a Docker image from Dockerfile, run this command on the main directory.**

```
docker build -t <image name> .
```

**2. Create and start a Docker container**

```
docker run -it -p <host port>:<container port> --name <container name> -v <host directory>:<container directory> etarho/dl-gpu:<version> /bin/bash
```

**3. Connect to the workstation server**

```
ssh -L <port(client)>:localhost:<port(host)> <user>@<host IP address>
```

**4. Run JupyterLab**

```
jupyter lab --allow-root
```

**5. Access to JupyterLab**

Access to `localhost:<port(client)>/lab` with your browser.

## How to analyze

**1. Visualize training results**

```
mlflow server -p <port> -h 0.0.0.0
```
Then, access to `localhost:<port>` with your browser.

**Example**

```
docker run -it -p 8888:8888 --gpus 0 --name dl -v "/home/rock/workspace:/home/workspace" etarho/dl-gpu:1.2 /bin/bash
```



**2. Hyperparameter Optimization**

```
optuna dashboard --storage 'sqlite:///<database>.db' --study-name '<study name>'
```

 