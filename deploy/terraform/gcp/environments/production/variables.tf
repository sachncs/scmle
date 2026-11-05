variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "image_digest" {
  description = "Digest of the release image (sha256:<64 hex>); passed by the release workflow."
  type        = string
}

variable "invoker_members" {
  type = list(string)
}

variable "jwt_issuer" {
  type = string
}

variable "jwt_audience" {
  type = string
}

variable "github_repository" {
  type = string
}

variable "billing_account" {
  type = string
}

variable "budget_emails" {
  type = list(string)
}
