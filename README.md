# Versich Treue: Vehicle Insurance Churn Prediction

An end-to-end MLOps pipeline for predicting vehicle insurance policy churn with automated CI/CD deployment.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Directory Structure](#directory-structure)
- [Local Development Setup](#local-development-setup)
- [Cloud Infrastructure Setup](#cloud-infrastructure-setup)
- [CI/CD Pipeline Setup](#cicd-pipeline-setup)
- [Usage](#usage)

## Features

- **End-to-End ML Pipeline**: Data ingestion → validation → transformation → training → evaluation
- **Database Integration**: MongoDB Atlas for scalable data storage
- **Artifact Management**: AWS S3 for storing trained models
- **Automated CI/CD**: GitHub Actions with Docker + ECR integration
- **Cloud Deployment**: AWS EC2 / ECS / EKS deployment
- **Containerization**: Portable builds using Docker
- **Prediction API**: FastAPI web service for inference

## Tech Stack

| Category         | Technologies                        |
|-----------------|-------------------------------------|
| Language         | Python 3.13                         |
| Data Management  | Pandas, MongoDB Atlas               |
| ML Framework     | Scikit-learn                        |
| Cloud Services   | AWS (S3, EC2, ECR, IAM)            |
| CI/CD            | GitHub Actions, Docker              |
| Web Framework    | FastAPI                             |
| Development      | virtualenv / pip                    |

## Directory Structure

```
versich-treue/
├── .dockerignore
├── .env.example
├── .github/
│   └── workflows/
│       └── aws.yaml
├── Dockerfile
├── notebook/
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── aws_storage/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_evaluation.py
│   │   └── model_trainer.py
│   ├── configuration/
│   ├── constants/
│   ├── data_access/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipeline/
│   └── utils/
├── static/
├── templates/
└── app.py
```

## Local Development Setup

1. **Clone & Initialize**

```bash
git clone <your-repo-url>
cd versich-treue
python3 template.py  # initialize project structure
```

2. **Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Environment Configuration**

Create `.env` in project root:

```env
MONGODB_URL="mongodb+srv://<username>:<password>@<cluster-url>/"
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
```

Load environment variables:

```bash
export $(cat .env | xargs)
```

## Cloud Infrastructure Setup

### MongoDB Atlas

* Create free-tier cluster
* Add secure DB user
* Whitelist required IPs
* Upload dataset using `notebook/mongoDB_demo.ipynb`

### AWS

* **IAM User**: Create with AdministratorAccess (least privilege recommended)
* **S3 Bucket**: For storing artifacts
* **EC2 Instance**: Ubuntu 24.04 LTS, t2.medium, 30GB storage
* **Security Group**: Allow HTTP (80), HTTPS (443), and app port (8080)

## CI/CD Pipeline Setup

* **ECR Repository**: Create private repo `versich-treue-ecr`
* **EC2 Runner Setup**

```bash
sudo apt update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

* Configure self-hosted runner with GitHub Actions.

* **GitHub Secrets**

  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `AWS_DEFAULT_REGION=us-east-1`
  * `ECR_REPO=versich-treue-ecr`

## Usage

### Training & Serving

```bash
python3 app.py
```

### Access locally

```
http://127.0.0.1:8080/
```

### Predictions

| Environment | URL                                            |
| ----------- | ---------------------------------------------- |
| Local       | [http://127.0.0.1:8080](http://127.0.0.1:8080) |
| Cloud       | http://<EC2_PUBLIC_IP>:8080                    |
