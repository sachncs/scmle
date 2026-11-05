output "gateway_url" {
  value = aws_apigatewayv2_api.this.api_endpoint
}

output "repository_url" {
  value = aws_ecr_repository.this.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "service_name" {
  value = aws_ecs_service.serve.name
}

output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "deploy_role_arn" {
  value = aws_iam_role.deploy.arn
}

output "train_role_arn" {
  value = aws_iam_role.train.arn
}

output "serve_role_arn" {
  value = aws_iam_role.serve.arn
}

output "kms_key_arn" {
  value = aws_kms_key.this.arn
}
