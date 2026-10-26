pipeline {
    agent {
        node {
            label 'local'
        }
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/hardattmangrola/cryptopulse.git'
            }
        }

        stage('Setup Environment') {
            steps {
                withCredentials([file(credentialsId: 'cryptopulse-env', variable: 'SECRET_ENV')]) {
                    bat 'copy /Y "%SECRET_ENV%" .env'
                }
            }
        }

        stage('Show Workspace') {
            steps {
                bat 'cd'
                bat 'docker --version'
                bat 'docker compose version'
            }
        }

        stage('Build Images') {
            steps {
                bat 'docker compose build'
            }
        }

        stage('Run Tests') {
            steps {
                bat 'pytest tests -v'
            }
        }

        stage('Deploy Containers') {
            steps {
                bat 'docker compose up -d'
            }
        }
    }

    post {
        success {
            echo 'CI/CD pipeline completed successfully.'
        }

        failure {
            echo 'Pipeline failed. Check the stage logs.'
        }
    }
}
