resource "aws_cloudwatch_log_group" "handler" {
  name              = "/aws/batch/football-perspectives-${var.env_name}"
  retention_in_days = var.log_retention_days
}
