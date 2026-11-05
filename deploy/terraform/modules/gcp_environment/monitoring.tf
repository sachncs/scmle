resource "google_monitoring_notification_channel" "email" {
  for_each = toset(var.budget_emails)

  display_name = "${local.name} ${each.value}"
  type         = "email"
  labels       = { email_address = each.value }

  depends_on = [google_project_service.this]
}

resource "google_monitoring_alert_policy" "errors" {
  display_name          = "${local.name} 5xx responses"
  combiner              = "OR"
  notification_channels = [for channel in google_monitoring_notification_channel.email : channel.id]
  user_labels           = local.labels

  conditions {
    display_name = "5xx count"

    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${google_cloud_run_v2_service.serve.name}\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      comparison      = "COMPARISON_GT"
      threshold_value = 5
      duration        = "300s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }
}

resource "google_monitoring_alert_policy" "latency" {
  display_name          = "${local.name} latency"
  combiner              = "OR"
  notification_channels = [for channel in google_monitoring_notification_channel.email : channel.id]
  user_labels           = local.labels

  conditions {
    display_name = "p95 above two seconds"

    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${google_cloud_run_v2_service.serve.name}\" AND metric.type = \"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 2000
      duration        = "300s"

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }
}

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "${local.name} monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_usd)
    }
  }

  threshold_rules {
    threshold_percent = 0.8
  }

  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  all_updates_rule {
    monitoring_notification_channels = [for channel in google_monitoring_notification_channel.email : channel.id]
    disable_default_iam_recipients   = false
  }
}
