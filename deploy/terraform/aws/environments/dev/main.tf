# AWS dev. Backend settings (bucket, key, region, lock table) are passed with
# -backend-config at init time; nothing about the state location or any
# credential is stored here.
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }

  backend "s3" {}
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      "scmle-environment" = "dev"
      "scmle-managed-by"  = "terraform"
    }
  }
}

module "environment" {
  source = "../../../modules/aws_environment"

  environment              = "dev"
  region                   = var.region
  image_digest             = var.image_digest
  jwt_issuer               = var.jwt_issuer
  jwt_audience             = var.jwt_audience
  github_repository        = var.github_repository
  github_oidc_provider_arn = var.github_oidc_provider_arn
  budget_emails            = var.budget_emails

  vpc_cidr            = "10.20.0.0/16"
  cpu                 = 256
  memory              = 512
  min_tasks           = 1
  max_tasks           = 2
  monthly_budget_usd  = 100
  destroyable         = true
  deletion_protection = false
}
