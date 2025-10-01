# Versich Treue: Vehicle Insurance Churn Prediction

This repository contains the source code for **Versich Treue**, an end-to-end machine learning project designed to predict vehicle insurance policy churn. The project implements a complete MLOps pipeline, from data ingestion and validation to automated model training, deployment, and CI/CD.

## Features
* **End-to-End ML Pipeline**: Implements Data Ingestion, Validation, Transformation, Model Training, and Evaluation.
* **Database Integration**: Uses MongoDB Atlas for storing and retrieving the dataset.
* **Cloud Storage**: Leverages AWS S3 for storing trained model artifacts.
* **Automated CI/CD**: Utilizes GitHub Actions for continuous integration and deployment.
* **Containerization**: Docker is used to containerize the application for consistent deployments.
* **Cloud Deployment**: The application is deployed on an AWS EC2 instance.
* **Web Interface**: A simple web application built with Flask for making predictions and triggering training.

## Tech Stack
* **Programming Language**: Python 3.13
* **Data Management**: Pandas, MongoDB
* **ML Framework**: Scikit-learn
* **Cloud Services**: AWS (S3, EC2, ECR, IAM)
* **CI/CD**: Docker, GitHub Actions (Self-hosted Runner)
* **Web Framework**: FAST API

## Project Workflow

1.  **Data Storage**: The initial dataset is stored in a MongoDB Atlas database.
2.  **Data Ingestion**: The pipeline fetches the data from MongoDB and stores it as a local artifact.
3.  **Data Validation**: Data is validated against a predefined schema (`schema.yaml`) to check for correctness, data types, and drift.
4.  **Data Transformation**: Preprocessing steps like feature engineering, scaling, and encoding are applied.
5.  **Model Training**: A machine learning model is trained on the transformed data.
6.  **Model Evaluation**: The trained model is evaluated against a metric threshold. If the new model is better, it is pushed to an AWS S3 bucket.
7.  **CI/CD Trigger**: A `git push` to the main branch triggers the GitHub Actions workflow.
8.  **Build & Push Image**: The workflow builds a Docker image of the application and pushes it to Amazon ECR.
9.  **Deployment**: A self-hosted runner on an AWS EC2 instance pulls the latest Docker image from ECR and runs the container, deploying the application.

## Directory Structure
.
├── .dockerignore
├── .env
├── .github
│   └── workflows
│       └── aws.yaml
├── .gitignore
├── Dockerfile
├── app.py
├── config/
│   ├── model.yaml
│   └── schema.yaml
├── notebook/
├── pyproject.toml
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── cloud_storage/
│   │   ├── __init__.py
│   │   └── aws_storage.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_deployment.py
│   │   ├── model_evaluation.py
│   │   └── model_training.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── aws_connection.py
│   │   └── mongo_db_connection.py
│   ├── constants/
│   │   └── __init__.py
│   ├── data_access/
│   │   ├── __init__.py
│   │   └── vt_data.py
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   ├── exception/
│   │   └── __init__.py
│   ├── logger/
│   │   └── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── main_utils.py
├── static/
├── templates/


***

## Local Development Setup

Follow these steps to set up the project on your local machine.

### 1. Project Initialization
Clone the repository and run the template script to create the necessary project structure.
```bash
git clone <your-repo-url>
cd versich-treue
python3 template.py
2. Create and Activate Virtual Environment
It is highly recommended to use a virtual environment.

Bash

python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Install all the required packages, including the local src package defined in setup.py.

Bash

pip install -r requirements.txt
After installation, you can verify that the local src package is installed by running pip list.

4. Environment Variables
Create a .env file in the root directory to store your sensitive credentials. Add the following variables:

MONGODB_URL="mongodb+srv://<username>:<password>@<cluster-url>/"
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
To load these variables into your shell session, you can use export $(cat .env | xargs).

Cloud Infrastructure Setup
1. MongoDB Atlas
Create Cluster: Sign up for MongoDB Atlas and create a free tier (M0) cluster.

Database User: In "Database Access," create a new database user with a secure password.

Network Access: In "Network Access," whitelist connections from anywhere by adding the IP address 0.0.0.0/0.

Connection String: Go to "Database," click "Connect," select "Drivers," and copy the Python connection string. Update the MONGODB_URL in your .env file with this string, replacing <password> with your user's password.

Push Data: Use a notebook (e.g., notebook/mongoDB_demo.ipynb) to load your initial dataset into the created MongoDB collection.

2. AWS Setup
IAM User:

In the AWS IAM console, create a new user.

Attach the AdministratorAccess policy.

Create an access key for this user (select CLI option) and download the CSV file.

Add the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to your .env file.

S3 Bucket:

In the AWS S3 console, create a new bucket (e.g., versich-treue-bucket) in the us-east-1 region.

Uncheck "Block all public access" and acknowledge the warning. This bucket will be used to store trained model artifacts.

CI/CD Pipeline Setup
This project uses GitHub Actions with a self-hosted runner on an EC2 instance for deployment.

1. ECR Repository
In the AWS ECR console, create a new private repository (e.g., versich-treue-ecr) in the us-east-1 region. This will store the Docker images.

2. EC2 Instance (Self-Hosted Runner)
Launch Instance:

Go to the EC2 console and launch a new instance.

AMI: Ubuntu Server 24.04 LTS (Free Tier eligible).

Instance Type: t2.medium (Note: This may incur charges).

Key Pair: Create and download a new key pair to access the instance later.

Network Settings: Allow HTTP and HTTPS traffic from the internet.

Storage: Increase storage to 30 GB.

Install Docker: Connect to your EC2 instance and install Docker.

Bash

sudo apt-get update -y
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
Configure Runner:

In your GitHub repository, go to Settings > Actions > Runners > New self-hosted runner.

Select Linux as the OS.

Follow the "Download" and "Configure" commands provided by GitHub on your EC2 terminal. When prompted, you can accept the default settings.

Start the runner by executing ./run.sh. The runner should now show as "Idle" in your GitHub settings.

3. GitHub Secrets
For the workflow to access your AWS account, you must add the following repository secrets in Settings > Secrets and variables > Actions:

AWS_ACCESS_KEY_ID: Your AWS access key ID.

AWS_SECRET_ACCESS_KEY: Your AWS secret access key.

AWS_DEFAULT_REGION: us-east-1.

ECR_REPO: The name of your ECR repository (e.g., versich-treue-ecr).

4. Expose Application Port
To access the web application, you need to open the port it runs on (e.g., 5080) in the EC2 instance's security group.

Go to your EC2 instance in the AWS console.

Navigate to the Security tab and click on the security group.

Click Edit inbound rules.

Add a new rule:

Type: Custom TCP

Port Range: 8080

Source: Anywhere (0.0.0.0/0)

Save the rules.

Usage
Running the Training Pipeline
You can trigger the entire training pipeline by running:

Bash

python3 app.py
Then, navigate to http://127.0.0.1:8080/ in your browser.

Making Predictions
Once the application is running (either locally or on EC2), you can access the home page to input data and get a churn prediction.

Local: http://127.0.0.1:8080

Deployed: http://<EC2_PUBLIC_IP_ADDRESS>:8080
