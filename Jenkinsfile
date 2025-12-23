pipeline {
    agent any
    
    environment {
        PROJECT_ID = 'unet-segmentation-482119'
        REGION = 'us-central1'
        SERVICE_NAME = 'unet-segmentation'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                sh '''
                    docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${BUILD_NUMBER} .
                    docker tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${BUILD_NUMBER} gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest
                '''
            }
        }
        
        stage('Push to GCR') {
            steps {
                echo 'Pushing to Google Container Registry...'
                withCredentials([file(credentialsId: 'gcp-credentials', variable: 'GCP_KEY')]) {
                    sh '''
                        gcloud auth activate-service-account --key-file=${GCP_KEY}
                        gcloud config set project ${PROJECT_ID}
                        gcloud auth configure-docker
                        docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${BUILD_NUMBER}
                        docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest
                    '''
                }
            }
        }
        
        stage('Deploy to Cloud Run') {
            steps {
                echo 'Deploying to Cloud Run...'
                withCredentials([file(credentialsId: 'gcp-credentials', variable: 'GCP_KEY')]) {
                    sh '''
                        gcloud run deploy ${SERVICE_NAME} \
                            --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${BUILD_NUMBER} \
                            --platform managed \
                            --region ${REGION} \
                            --allow-unauthenticated \
                            --memory 2Gi \
                            --timeout 300
                    '''
                }
            }
        }
    }
    
    post {
        success {
            echo 'Deployment successful!'
        }
        failure {
            echo 'Deployment failed!'
        }
    }
}
