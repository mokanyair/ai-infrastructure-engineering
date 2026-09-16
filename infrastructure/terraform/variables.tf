variable "aws_region" {
  description = "AWS Region for the inference lab."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type for the experiment."
  type        = string
  default     = "m6i.xlarge"

  validation {
    condition = contains(
      ["m6i.xlarge", "m6i.2xlarge"],
      var.instance_type
    )

    error_message = "Only the approved baseline and scaling instance types are allowed."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for the lab VPC."
  type        = string
  default     = "10.60.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block for the public subnet."
  type        = string
  default     = "10.60.1.0/24"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GiB."
  type        = number
  default     = 40

  validation {
    condition     = var.root_volume_size >= 40
    error_message = "The root volume must be at least 40 GiB."
  }
}

variable "ami_id" {
  description = "Optional pinned Amazon Linux 2023 x86_64 AMI ID. Required for repeat experiments."
  type        = string
  default     = null
}

variable "common_tags" {
  description = "Common resource tags."
  type        = map(string)

  default = {
    Project     = "AI-Infrastructure-Lab"
    Week        = "01"
    Environment = "Training"
    ManagedBy   = "Terraform"
    CostCenter  = "Personal-Lab"
    Expiry      = "2026-09-20"
  }
}

