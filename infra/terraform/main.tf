terraform {
  required_version = ">= 1.7.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.30.0, < 3.0.0"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

resource "kubernetes_namespace_v1" "fleet" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_persistent_volume_claim_v1" "eta_models" {
  metadata {
    name      = var.model_pvc_name
    namespace = kubernetes_namespace_v1.fleet.metadata[0].name
  }

  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = var.model_volume_size
      }
    }
  }
}

resource "kubernetes_deployment_v1" "eta_api" {
  metadata {
    name      = "fleet-eta-api"
    namespace = kubernetes_namespace_v1.fleet.metadata[0].name
    labels = {
      app = "fleet-eta-api"
    }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        app = "fleet-eta-api"
      }
    }

    template {
      metadata {
        labels = {
          app = "fleet-eta-api"
        }
      }

      spec {
        security_context {
          run_as_non_root = true
          seccomp_profile {
            type = "RuntimeDefault"
          }
        }

        container {
          name              = "api"
          image             = var.image
          image_pull_policy = "IfNotPresent"

          port {
            name           = "http"
            container_port = 8000
          }

          env {
            name  = "ETA_MODEL_PATH"
            value = "/models/eta_hist_gradient_boosting.joblib"
          }

          volume_mount {
            name       = "eta-model"
            mount_path = "/models"
            read_only  = true
          }

          readiness_probe {
            http_get {
              path = "/ready"
              port = "http"
            }
            initial_delay_seconds = 3
            period_seconds        = 10
            failure_threshold     = 3
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 10
            period_seconds        = 20
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "1"
              memory = "1Gi"
            }
          }

          security_context {
            allow_privilege_escalation = false
            read_only_root_filesystem  = true
            capabilities {
              drop = ["ALL"]
            }
          }
        }

        volume {
          name = "eta-model"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim_v1.eta_models.metadata[0].name
            read_only  = true
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "eta_api" {
  metadata {
    name      = "fleet-eta-api"
    namespace = kubernetes_namespace_v1.fleet.metadata[0].name
  }

  spec {
    selector = {
      app = "fleet-eta-api"
    }

    port {
      name        = "http"
      port        = 80
      target_port = "http"
    }

    type = "ClusterIP"
  }
}
