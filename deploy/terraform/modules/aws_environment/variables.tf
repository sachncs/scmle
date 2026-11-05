variable "environment" {
  description = "Environment name; it is also the value of the scmle-environment tag."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be dev, staging or production."
  }
}

variable "name_prefix" {
  description = "Prefix of every resource name."
  type        = string
  default     = "scmle"
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
  description = "Fargate task CPU units."
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 1024
}

variable "min_tasks" {
  type    = number
  default = 1
}

variable "max_tasks" {
  type    = number
  default = 4
}

variable "target_cpu_percent" {
  description = "Average CPU the autoscaler holds."
  type        = number
  default     = 60
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "jwt_issuer" {
  description = "Issuer of the bearer tokens the gateway accepts."
  type        = string
}

variable "jwt_audience" {
  description = "Audience of the bearer tokens the gateway accepts."
  type        = string
}

variable "gateway_rate_limit" {
  description = "Steady requests per second the gateway admits."
  type        = number
  default     = 50
}

variable "gateway_burst_limit" {
  type    = number
  default = 100
}

variable "github_repository" {
  description = "owner/name of the repository whose workflows may deploy."
  type        = string
}

variable "github_oidc_provider_arn" {
  description = "ARN of the account's GitHub OIDC identity provider (created once per account)."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget in US dollars."
  type        = number

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd must be positive."
  }
}

variable "budget_emails" {
  description = "Recipients of budget and alarm notifications."
  type        = list(string)

  validation {
    condition     = length(var.budget_emails) > 0
    error_message = "at least one budget e-mail is required."
  }
}

variable "log_retention_days" {
  type    = number
  default = 30
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

variable "region" {
  type = string
}
