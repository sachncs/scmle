variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "environment" {
  description = "Environment name; it is also the value of the scmle-environment label."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be dev, staging or production."
  }
}

variable "name_prefix" {
  type    = string
  default = "scmle"
}

variable "image_digest" {
  description = "Digest of the service image (sha256:<64 hex>). A tag is refused."
  type        = string

  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "image_digest must be sha256: followed by 64 lowercase hex characters."
  }
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  type    = string
  default = "1Gi"
}

variable "min_instances" {
  type    = number
  default = 1
}

variable "max_instances" {
  type    = number
  default = 4
}

variable "concurrency" {
  description = "Concurrent requests per instance."
  type        = number
  default     = 20
}

variable "invoker_members" {
  description = "Principals allowed to call the service (for example serviceAccount:... or group:...). Never allUsers."
  type        = list(string)

  validation {
    condition     = length(var.invoker_members) > 0 && !contains(var.invoker_members, "allUsers") && !contains(var.invoker_members, "allAuthenticatedUsers")
    error_message = "invoker_members must be non-empty and must not contain allUsers or allAuthenticatedUsers."
  }
}

variable "jwt_issuer" {
  type = string
}

variable "jwt_audience" {
  type = string
}

variable "github_repository" {
  description = "owner/name of the repository whose workflows may deploy."
  type        = string
}

variable "billing_account" {
  description = "Billing account the budget is attached to."
  type        = string
}

variable "monthly_budget_usd" {
  type = number

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd must be positive."
  }
}

variable "budget_emails" {
  type = list(string)

  validation {
    condition     = length(var.budget_emails) > 0
    error_message = "at least one budget e-mail is required."
  }
}

variable "secret_names" {
  description = "Names of the secrets the service reads. Only the containers are created; values are set out of band."
  type        = list(string)
  default     = ["signing-key", "auth-keys"]
}

variable "destroyable" {
  description = "Whether a destroy may remove stateful resources at once (dev and staging only)."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  type    = bool
  default = true
}
