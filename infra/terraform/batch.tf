resource "aws_batch_compute_environment" "gpu_spot" {
  compute_environment_name_prefix = "football-perspectives-${var.env_name}-"
  type                            = "MANAGED"
  state                           = "ENABLED"
  service_role                    = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    bid_percentage      = var.spot_bid_percentage
    spot_iam_fleet_role = aws_iam_role.spot_fleet.arn

    min_vcpus     = 0
    desired_vcpus = 0
    max_vcpus     = var.max_vcpus

    instance_type = var.instance_types
    instance_role = aws_iam_instance_profile.instance.arn

    subnets            = aws_subnet.public[*].id
    security_group_ids = [aws_security_group.batch.id]

    tags = {
      Name = "football-perspectives-${var.env_name}-gpu-worker"
    }
  }

  # Recreating in-place is cheap when min/desired are 0, and avoids the
  # "compute_environment is enabled, can't modify in place" error pattern.
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_batch_job_queue" "hmr_world" {
  name     = "football-perspectives-${var.env_name}-hmr-world"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_spot.arn
  }
}

resource "aws_batch_job_definition" "hmr_world" {
  name                  = "football-perspectives-${var.env_name}-hmr-world"
  type                  = "container"
  platform_capabilities = ["EC2"]

  timeout {
    attempt_duration_seconds = var.job_attempt_duration_seconds
  }

  retry_strategy {
    attempts = var.job_attempts

    evaluate_on_exit {
      # Spot interruption — retry.
      on_status_reason = "Host EC2*"
      action           = "RETRY"
    }

    evaluate_on_exit {
      # Anything else — don't waste retries on application bugs.
      on_status_reason = "*"
      action           = "EXIT"
    }
  }

  container_properties = jsonencode({
    image       = "${aws_ecr_repository.handler.repository_url}:${var.image_tag}"
    jobRoleArn  = aws_iam_role.job.arn
    vcpus       = 4
    memory      = 15000
    resourceRequirements = [
      { type = "GPU", value = "1" },
    ]
    environment = [
      { name = "PYTHONUNBUFFERED", value = "1" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.handler.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "hmr-world"
      }
    }
  })
}
