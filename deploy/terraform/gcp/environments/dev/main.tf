# GCP dev. The state bucket and prefix are passed with -backend-config at
# init time; nothing about the state location or any credential is stored here.
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.10"
    }
  }

  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "environment" {
  source = "../../../modules/gcp_environment"

  environment       = "dev"
  project_id        = var.project_id
  region            = var.region
  image_digest      = var.image_digest
  invoker_members   = var.invoker_members
  jwt_issuer        = var.jwt_issuer
  jwt_audience      = var.jwt_audience
  github_repository = var.github_repository
  billing_account   = var.billing_account
  budget_emails     = var.budget_emails

  cpu                 = "1"
  memory              = "512Mi"
  min_instances       = 0
  max_instances       = 2
  monthly_budget_usd  = 100
  destroyable         = true
  deletion_protection = false
}
