output "region" {
  value = var.region
}

output "s3_bucket" {
  description = "Name of the runs bucket — feed into HMR_WORLD_S3_BUCKET."
  value       = aws_s3_bucket.runs.bucket
}

output "ecr_repo_url" {
  description = "ECR repository URL — push the handler image here."
  value       = aws_ecr_repository.handler.repository_url
}

output "batch_job_queue" {
  description = "Batch job queue ARN — feed into HMR_WORLD_BATCH_JOB_QUEUE."
  value       = aws_batch_job_queue.hmr_world.arn
}

output "batch_job_definition" {
  description = "Batch job definition ARN — feed into HMR_WORLD_BATCH_JOB_DEFINITION."
  value       = aws_batch_job_definition.hmr_world.arn
}

output "cloudwatch_log_group" {
  description = "Where handler stderr ends up."
  value       = aws_cloudwatch_log_group.handler.name
}
