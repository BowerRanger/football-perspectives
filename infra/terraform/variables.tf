variable "region" {
  description = "AWS region. us-east-1 is the default; revisit if Spot capacity becomes a problem."
  type        = string
  default     = "us-east-1"
}

variable "env_name" {
  description = "Logical environment name (prod, dev, etc.) — drives bucket/repo naming."
  type        = string
  default     = "prod"
}

variable "s3_bucket_name" {
  description = "Globally-unique S3 bucket name. Convention: football-perspectives-hmr-{env}-{account-id}."
  type        = string
}

variable "ecr_repo_name" {
  description = "ECR repository name for the handler image."
  type        = string
  default     = "football-perspectives-hmr-world"
}

variable "max_vcpus" {
  description = "Cap on concurrent vCPUs in the compute environment. 96 = 24 jobs × 4 vCPU."
  type        = number
  default     = 96
}

variable "instance_types" {
  description = "EC2 instance types Batch may launch. g4dn.xlarge gives one T4. Add g4dn.2xlarge / g5.xlarge for Spot capacity diversity."
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "image_tag" {
  description = "Container image tag (set by CI to the git SHA on push)."
  type        = string
  default     = "latest"
}

variable "job_attempt_duration_seconds" {
  description = "Hard timeout per job attempt. 600s is 10× the expected ~60s/track."
  type        = number
  default     = 600
}

variable "job_attempts" {
  description = "Retry attempts per array element. 2 covers Spot interruptions."
  type        = number
  default     = 2
}

variable "log_retention_days" {
  description = "CloudWatch log group retention."
  type        = number
  default     = 30
}

variable "s3_lifecycle_days" {
  description = "Per-run S3 prefix lifetime. Outputs are pulled back to local disk by the orchestrator."
  type        = number
  default     = 30
}

variable "spot_bid_percentage" {
  description = "Max Spot price as percentage of on-demand. 100 = let Batch use on-demand price as the ceiling."
  type        = number
  default     = 100
}
