variable "kubeconfig_path" {
  description = "Path to the kubeconfig used by the Kubernetes provider."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Optional kubeconfig context."
  type        = string
  default     = null
  nullable    = true
}

variable "namespace" {
  description = "Kubernetes namespace for fleet intelligence workloads."
  type        = string
  default     = "fleet-intelligence"
}

variable "image" {
  description = "Container image containing the v0.5 API."
  type        = string
  default     = "ghcr.io/zubairz4far/production-fleet-intelligence-platform:v0.5.0"
}

variable "replicas" {
  description = "ETA API replica count."
  type        = number
  default     = 2

  validation {
    condition     = var.replicas >= 1
    error_message = "replicas must be at least 1"
  }
}

variable "model_pvc_name" {
  description = "PVC containing eta_hist_gradient_boosting.joblib."
  type        = string
  default     = "fleet-eta-models"
}

variable "model_volume_size" {
  description = "Requested model PVC size."
  type        = string
  default     = "1Gi"
}
