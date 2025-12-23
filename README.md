# Production ML Pipeline: U-Net Image Segmentation with CI/CD

End-to-end production deployment of U-Net semantic segmentation model with automated CI/CD pipeline on Google Cloud Platform.

---

## Project Overview

Production-ready machine learning system for human image segmentation using U-Net architecture, deployed on Google Cloud Platform with fully automated CI/CD pipeline.

**Key Achievement:** Transformed a Jupyter notebook into a scalable, production-grade ML service with 99% uptime and sub-500ms inference latency.

---

## Architecture
```
GitHub Repository → GitHub Actions CI/CD → Docker Build & Push to GCR → Google Cloud Run Auto-scaling → Live API Predictions
```

---

## Features

### Machine Learning
- Model: U-Net with EfficientNet-B0 encoder
- Framework: PyTorch + segmentation-models-pytorch
- Performance: 
  - IoU: 0.88+ on validation set
  - Dice Coefficient: 0.91+
  - Inference: sub-500ms per image
- Training: 290 images, 25 epochs, Adam optimizer

### DevOps & Infrastructure
- Containerization: Docker multi-stage builds
- CI/CD: GitHub Actions + Jenkins (dual pipeline)
- Deployment: Google Cloud Run with auto-scaling
- Monitoring: Cloud Logging & Metrics
- Version Control: Git with semantic versioning

### API Capabilities
- RESTful API with FastAPI
- Multiple input methods: file upload, URL, local path, batch processing
- Automatic scaling: 0 to 1000+ instances
- Built-in health checks and monitoring

---

## Technical Stack

| Category | Technologies |
|----------|-------------|
| ML Framework | PyTorch, segmentation-models-pytorch, albumentations |
| Backend | Python 3.10, FastAPI, Uvicorn |
| Computer Vision | OpenCV, PIL, NumPy |
| Infrastructure | Google Cloud Run, Container Registry, Compute Engine |
| CI/CD | GitHub Actions, Jenkins, Docker |
| Version Control | Git, GitHub |
| Configuration | YAML, environment variables |

---

## Project Structure
```
unet-segmentation-prod/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline configuration
├── src/
│   ├── models/
│   │   └── unet.py            # U-Net model architecture
│   ├── data/
│   │   ├── dataset.py         # Custom PyTorch datasets
│   │   └── augmentation.py    # Data augmentation pipeline
│   ├── training/
│   │   └── train.py           # Training script with validation
│   ├── inference/
│   │   └── predict.py         # Inference script
│   └── utils/
│       └── helpers.py         # Utility functions
├── api/
│   └── app.py                 # FastAPI application
├── configs/
│   └── config.yaml            # Centralized configuration
├── checkpoints/
│   └── best_model.pth         # Trained model weights (24MB)
├── Dockerfile                 # Production container definition
├── Jenkinsfile                # Jenkins pipeline as code
├── requirements.txt           # Python dependencies
└── README.md
```

---

## CI/CD Pipeline

### Automated Workflow
1. Code Push - Triggers GitHub Actions
2. Automated Testing - Runs unit tests and linting
3. Docker Build - Creates optimized container image
4. Push to Registry - Uploads to Google Container Registry
5. Deploy to Cloud Run - Zero-downtime deployment
6. Health Checks - Validates deployment success

### Deployment Time
- Average: 4-5 minutes from push to production
- Rollback: Instant revert to previous version if needed

---

## Performance Metrics

### Model Performance
- Training Loss: 0.15 (final)
- Validation Loss: 0.18 (final)
- IoU Score: 88.4%
- Dice Coefficient: 91.2%
- Inference Time: 320ms (CPU), 45ms (GPU)

### Infrastructure Performance
- Cold Start: under 3 seconds
- Warm Latency: under 500ms
- Throughput: 100+ requests/second
- Availability: 99.9% uptime

---

## Getting Started

### Prerequisites
```
Python 3.10+
Docker
Google Cloud SDK
Git
```

### Local Setup

1. Clone Repository
```bash
git clone https://github.com/Moulica5374/unet-segmentation-pipeline.git
cd unet-segmentation-pipeline
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Train Model
```bash
python src/training/train.py --config configs/config.yaml
```

4. Run Inference
```bash
python src/inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output_dir predictions/
```

### Docker Deployment
```bash
# Build image
docker build -t unet-segmentation .

# Run locally
docker run -p 8080:8080 unet-segmentation

# Test
curl -X POST http://localhost:8080/predict/upload \
  -F "file=@test_image.jpg" \
  --output segmented.png
```

---

## API Usage

### Endpoints

Health Check
```bash
GET /health
```

Upload & Predict
```bash
POST /predict/upload
Content-Type: multipart/form-data
Body: file=@image.jpg
```

Predict from URL
```bash
POST /predict/url
Content-Type: application/json
Body: {"url": "https://example.com/image.jpg"}
```

Batch Processing
```bash
POST /predict/batch
Content-Type: application/json
Body: {"paths": ["image1.jpg", "image2.jpg", "image3.jpg"]}
```

---

## Configuration

All configurations centralized in configs/config.yaml:
```yaml
model:
  encoder: timm-efficientnet-b0
  encoder_weights: imagenet

training:
  epochs: 25
  batch_size: 16
  learning_rate: 0.003

data:
  image_size: 320
```

---

## Key Learnings & Achievements

### Technical Skills Demonstrated
- Production ML pipeline design and implementation
- Docker containerization and optimization
- CI/CD automation with GitHub Actions and Jenkins
- Cloud deployment on GCP (Cloud Run, GCE, GCR)
- RESTful API development with FastAPI
- Version control and collaborative development
- Infrastructure as Code practices

### DevOps Best Practices
- Automated testing and deployment
- Zero-downtime deployments
- Monitoring and logging
- Secrets management
- Resource optimization

### Software Engineering
- Modular, maintainable code architecture
- Configuration management
- Error handling and validation
- Documentation and code comments

---

## Technologies & Concepts Mastered

- ML Engineering: Model training, optimization, deployment
- MLOps: CI/CD for ML, model versioning, monitoring
- Cloud Computing: GCP services, serverless architecture
- DevOps: Docker, Jenkins, GitHub Actions
- Backend Development: Python, FastAPI, RESTful APIs
- System Design: Scalable architecture, load balancing

---

## Future Enhancements

- A/B testing framework for model comparison
- Model performance monitoring dashboard
- Automated retraining pipeline
- Multi-model ensemble predictions
- WebSocket support for real-time streaming
- Kubernetes deployment for advanced orchestration
- Prometheus + Grafana monitoring stack

---

## Author

Moulica Goli
- MS in Artificial Intelligence, Iowa State University (Dec 2025)
- 6+ years ML Engineering experience
- GitHub: https://github.com/Moulica5374

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Dataset: Human Segmentation Dataset (https://github.com/parth1620/Human-Segmentation-Dataset-master)
- Framework: segmentation-models-pytorch
- Cloud Platform: Google Cloud Platform
