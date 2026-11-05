# Cloud Run authenticates every request at Google's front end: only a
# principal with roles/run.invoker reaches the container, and the service
# verifies its bearer token again. No allUsers binding exists.
resource "google_cloud_run_v2_service" "serve" {
  name                = "${local.name}-serve"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection
  labels              = local.labels

  template {
    service_account                  = google_service_account.serve.email
    timeout                          = "30s"
    max_instance_request_concurrency = var.concurrency
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    vpc_access {
      egress = "PRIVATE_RANGES_ONLY"

      network_interfaces {
        network    = google_compute_network.this.id
        subnetwork = google_compute_subnetwork.this.id
      }
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.this.repository_id}/serve@${var.image_digest}"

      ports {
        container_port = var.container_port
      }

      resources {
        cpu_idle = true

        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      env {
        name  = "SCMLE_ENV"
        value = var.environment
      }

      env {
        name  = "SCMLE_AUTH_ISSUER"
        value = var.jwt_issuer
      }

      env {
        name  = "SCMLE_AUTH_AUDIENCE"
        value = var.jwt_audience
      }

      dynamic "env" {
        for_each = google_secret_manager_secret.this

        content {
          name = upper(replace(env.key, "-", "_"))

          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/readyz"
        }

        period_seconds    = 5
        failure_threshold = 12
      }

      liveness_probe {
        http_get {
          path = "/livez"
        }

        period_seconds = 30
      }

      volume_mounts {
        name       = "tmp"
        mount_path = "/tmp"
      }
    }

    volumes {
      name = "tmp"

      empty_dir {
        medium     = "MEMORY"
        size_limit = "64Mi"
      }
    }
  }

  # Releases deploy a new revision and the control plane shifts traffic; the
  # image and the traffic split are theirs to change, not Terraform's.
  lifecycle {
    ignore_changes = [client, client_version, traffic, template[0].containers[0].image, template[0].revision]
  }

  depends_on = [google_project_service.this]
}
