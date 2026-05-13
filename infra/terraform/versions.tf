terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "football-perspectives"
      Stage     = "hmr_world"
      ManagedBy = "terraform"
      Env       = var.env_name
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
