output "service_url" {
  value = module.environment.service_url
}

output "service_name" {
  value = module.environment.service_name
}

output "repository" {
  value = module.environment.repository
}

output "deploy_service_account" {
  value = module.environment.deploy_service_account
}

output "workload_identity_provider" {
  value = module.environment.workload_identity_provider
}
