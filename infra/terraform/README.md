# Terraform — AWS Batch infra for `hmr_world`

This module provisions everything `recon.py run --stages hmr_world`
needs when `hmr_world.runner = batch`:

- VPC + 2 public subnets + IGW (no NAT — Spot workers talk to S3/ECR over IGW)
- S3 bucket for per-run inputs/outputs (30-day expiration)
- ECR repository for the handler image
- IAM roles: Batch service, EC2 instance, Spot fleet, container task
- Batch managed compute environment on Spot (`g4dn.xlarge` by default)
- Batch job queue + job definition (with retry-on-Spot-interruption)
- CloudWatch log group with 30-day retention

## First-time bring-up

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars (set s3_bucket_name to a globally-unique name)

terraform init
terraform plan
terraform apply

# Push env vars into the shell so recon.py can find the Batch resources.
source ../../scripts/terraform_env.sh
```

`scripts/terraform_env.sh` reads `terraform output -json` and exports
`HMR_WORLD_S3_BUCKET`, `HMR_WORLD_BATCH_JOB_QUEUE`,
`HMR_WORLD_BATCH_JOB_DEFINITION` (the names the `${VAR}` references in
`config/default.yaml` resolve against).

## Pushing a new container image

```bash
# At repo root:
make build-image push-image       # see scripts/build_image.sh
# Update the job definition revision to point at the new SHA tag:
terraform apply -var "image_tag=$(git rev-parse --short HEAD)"
```

Or leave `image_tag = "latest"` and rely on the mutable tag while
iterating; pin to the SHA before any production-shaped run.

## Cost guardrails

| Knob                            | Default | Where to change                    |
| ------------------------------- | ------- | ---------------------------------- |
| max concurrent vCPUs            | 96      | `var.max_vcpus`                    |
| per-attempt timeout (s)         | 600     | `var.job_attempt_duration_seconds` |
| retry attempts                  | 2       | `var.job_attempts`                 |
| Spot bid % of on-demand         | 100     | `var.spot_bid_percentage`          |
| S3 retention on `runs/` (days)  | 30      | `var.s3_lifecycle_days`            |
| CloudWatch log retention (days) | 30      | `var.log_retention_days`           |
| ECR images kept                 | 20      | `ecr.tf` lifecycle rule            |

## Teardown

```bash
# Drain the bucket first (lifecycle policy doesn't help here).
aws s3 rm s3://${HMR_WORLD_S3_BUCKET} --recursive
terraform destroy
```
