output "service_url" {
  value = google_cloud_run_v2_service.serve.uri
}

output "service_name" {
  value = google_cloud_run_v2_service.serve.name
}

output "repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.this.repository_id}"
}

output "data_bucket" {
  value = google_storage_bucket.data.name
}

output "deploy_service_account" {
  value = google_service_account.deploy.email
}

output "train_service_account" {
  value = google_service_account.train.email
}

output "serve_service_account" {
  value = google_service_account.serve.email
}

output "workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}
