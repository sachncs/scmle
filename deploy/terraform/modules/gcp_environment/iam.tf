# Separate identities. serve is the running service, train is for training
# jobs, deploy is what the release workflow impersonates through workload
# identity federation (no key file exists). Every grant is on one resource;
# no primitive role and no project-wide admin role is used.

resource "google_service_account" "serve" {
  account_id   = "${local.name}-serve"
  display_name = "${local.name} serving"
}

resource "google_service_account" "train" {
  account_id   = "${local.name}-train"
  display_name = "${local.name} training"
}

resource "google_service_account" "deploy" {
  account_id   = "${local.name}-deploy"
  display_name = "${local.name} release workflow"
}

resource "google_storage_bucket_iam_member" "serve_models" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.serve.email}"

  condition {
    title      = "models only"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.data.name}/objects/models/\")"
  }
}

resource "google_storage_bucket_iam_member" "train_datasets" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.train.email}"

  condition {
    title      = "datasets only"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.data.name}/objects/datasets/\")"
  }
}

resource "google_storage_bucket_iam_member" "train_results" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.train.email}"

  condition {
    title      = "runs and models only"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.data.name}/objects/runs/\") || resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.data.name}/objects/models/\")"
  }
}

resource "google_secret_manager_secret_iam_member" "serve" {
  for_each = google_secret_manager_secret.this

  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.serve.email}"
}

resource "google_artifact_registry_repository_iam_member" "deploy" {
  repository = google_artifact_registry_repository.this.name
  location   = google_artifact_registry_repository.this.location
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deploy.email}"
}

resource "google_artifact_registry_repository_iam_member" "serve" {
  repository = google_artifact_registry_repository.this.name
  location   = google_artifact_registry_repository.this.location
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.serve.email}"
}

resource "google_cloud_run_v2_service_iam_member" "deploy" {
  name     = google_cloud_run_v2_service.serve.name
  location = google_cloud_run_v2_service.serve.location
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deploy.email}"
}

resource "google_service_account_iam_member" "deploy_acts_as_serve" {
  service_account_id = google_service_account.serve.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deploy.email}"
}

resource "google_cloud_run_v2_service_iam_member" "invoker" {
  for_each = toset(var.invoker_members)

  name     = google_cloud_run_v2_service.serve.name
  location = google_cloud_run_v2_service.serve.location
  role     = "roles/run.invoker"
  member   = each.value
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "${local.name}-github"
  display_name              = "${local.name} GitHub"

  depends_on = [google_project_service.this]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  attribute_condition                = "assertion.repository == \"${var.github_repository}\" && assertion.environment == \"${var.environment}\""

  attribute_mapping = {
    "google.subject"        = "assertion.sub"
    "attribute.repository"  = "assertion.repository"
    "attribute.environment" = "assertion.environment"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "deploy_federation" {
  service_account_id = google_service_account.deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}
