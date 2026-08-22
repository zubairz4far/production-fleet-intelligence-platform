output "namespace" {
  description = "Namespace containing the deployment."
  value       = kubernetes_namespace_v1.fleet.metadata[0].name
}

output "service_name" {
  description = "ClusterIP service name."
  value       = kubernetes_service_v1.eta_api.metadata[0].name
}

output "model_pvc_name" {
  description = "PVC that must contain the promoted ETA artifact before readiness passes."
  value       = kubernetes_persistent_volume_claim_v1.eta_models.metadata[0].name
}
