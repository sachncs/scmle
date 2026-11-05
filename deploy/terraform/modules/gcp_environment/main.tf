data "google_project" "this" {
  project_id = var.project_id
}

locals {
  name = "${var.name_prefix}-${var.environment}"
  labels = {
    "scmle-environment" = var.environment
    "scmle-managed-by"  = "terraform"
  }
  services = [
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudkms.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ]
}

resource "google_project_service" "this" {
  for_each = toset(local.services)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "this" {
  name                    = local.name
  auto_create_subnetworks = false

  depends_on = [google_project_service.this]
}

resource "google_compute_subnetwork" "this" {
  name                     = local.name
  region                   = var.region
  network                  = google_compute_network.this.id
  ip_cidr_range            = "10.30.0.0/24"
  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_kms_key_ring" "this" {
  name     = local.name
  location = var.region

  depends_on = [google_project_service.this]
}

resource "google_kms_crypto_key" "data" {
  name            = "data"
  key_ring        = google_kms_key_ring.this.id
  rotation_period = "7776000s"
  labels          = local.labels

  lifecycle {
    prevent_destroy = false
  }
}

data "google_storage_project_service_account" "this" {
  project = var.project_id
}

resource "google_kms_crypto_key_iam_member" "storage" {
  crypto_key_id = google_kms_crypto_key.data.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${data.google_storage_project_service_account.this.email_address}"
}

resource "google_storage_bucket" "data" {
  name                        = "${local.name}-${data.google_project.this.number}-data"
  location                    = var.region
  force_destroy               = var.destroyable
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  labels                      = local.labels

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.data.id
  }

  depends_on = [google_kms_crypto_key_iam_member.storage]
}

resource "google_artifact_registry_repository" "this" {
  repository_id = local.name
  location      = var.region
  format        = "DOCKER"
  labels        = local.labels

  docker_config {
    immutable_tags = true
  }

  depends_on = [google_project_service.this]
}

resource "google_secret_manager_secret" "this" {
  for_each = toset(var.secret_names)

  secret_id = "${local.name}-${each.value}"
  labels    = local.labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.this]
}
