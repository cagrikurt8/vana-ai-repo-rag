# Test Task for Querying Vana Project

This repository is built for connecting to Vanna project and querying the project information and codebase.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)

## Prerequisites

Make sure you have the following tools installed:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Connected to EPAM VPN via GlobalProtect.

## Setup

1. Create a .env file inside the repo with the help of the .env.example file. Fill in the necessary credentials. GITHUB_ACCESS_TOKEN is used for getting Vana repo content. It is not necessary to run the application.

2. Make sure Docker Desktop is running.

3. Create the docker image with the following command:
    ```sh
    docker build -t mbtiitsm-test-task:latest .
    ```

4. Start the container and map the ports to access it locally:
    ```sh
    docker run -p 8000:8000 mbtiitsm-test-task:latest
    ````

5. Test the application with the 'http://localhost:8000' address. To ask questions, you should send POST requests to the 'http://localhost:8000/ask' address with a JSON body:
```json
{
    "question": "What is Vanna AI?"
}
```