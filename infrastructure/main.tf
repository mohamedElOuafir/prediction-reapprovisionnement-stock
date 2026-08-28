provider "aws" {
  region = "eu-west-3"
}


resource "aws_s3_bucket" "data_bucket" {
  bucket = "reapprovisionnement-data-bucket"
  
}


resource "aws_s3_bucket" "models_bucket" {
  bucket = "reapprovisionnement-models-bucket"
}


resource "aws_s3_bucket_versioning" "models_versioning" {
  bucket = aws_s3_bucket.models_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}


resource "aws_ecr_repository" "reappro_api_repo" {
  name = "reapprovisionnement_api_repo"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}


resource "aws_iam_role" "lambda_exec_role" {
  name = "reapprovisionnement_lambda_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = { Service = "lambda.amazonaws.com"}
    }]
  })
}


resource "aws_iam_role_policy" "lambda_s3_read" {
  name = "reapprovisionnement_lambda_s3_read"
  role = aws_iam_role.lambda_exec_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.data_bucket.arn,
          aws_s3_bucket.models_bucket.arn,
          "${aws_s3_bucket.models_bucket.arn}/*", 
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
    }]
  })
}


resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}



resource "aws_lambda_function" "reapprovisionnement_api_function" {
  function_name = "reapprovisionnement_ml_api_function"
  role = aws_iam_role.lambda_exec_role.arn
  package_type = "Image"
  image_uri =  "${aws_ecr_repository.reappro_api_repo.repository_url}:latest"
  memory_size = 1024
  timeout = 30

  environment {
    variables = {
      S3_BUCKET_DATA = aws_s3_bucket.data_bucket.bucket
      S3_BUCKET_MODELS = aws_s3_bucket.models_bucket.bucket
    }
  }

  lifecycle {
    ignore_changes = [ image_uri ]
  }
}