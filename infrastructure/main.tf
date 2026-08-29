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


resource "aws_apigatewayv2_api" "http_api_gateway" {
  name = "reapprovisionnement_api_gateway"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_headers = ["Content-Type", "Accept"]
    allow_methods = ["GET", "POST", "OPTIONS"]
  }
}


resource "aws_apigatewayv2_integration" "reapp_lambda_integration" {
  api_id = aws_apigatewayv2_api.http_api_gateway.id
  integration_type = "AWS_PROXY"
  integration_uri = aws_lambda_function.reapprovisionnement_api_function.invoke_arn
  payload_format_version = "2.0"
}


resource "aws_apigatewayv2_route" "check_health_route" {
  api_id = aws_apigatewayv2_api.http_api_gateway.id
  route_key = "GET /health_check"
  target = "integrations/${aws_apigatewayv2_integration.reapp_lambda_integration.id}"
}


resource "aws_apigatewayv2_route" "prediction_route" {
  api_id    = aws_apigatewayv2_api.http_api_gateway.id
  route_key = "POST /prediction"
  target    = "integrations/${aws_apigatewayv2_integration.reapp_lambda_integration.id}"
}


resource "aws_apigatewayv2_stage" "prod_stage" {
  api_id      = aws_apigatewayv2_api.http_api_gateway.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 50
    throttling_rate_limit  = 20
  }
}

resource "aws_lambda_permission" "api_gateway_permission" {
  statement_id = "AllowExecutionFromApiGateway"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.reapprovisionnement_api_function.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.http_api_gateway.execution_arn}/*/*"
}