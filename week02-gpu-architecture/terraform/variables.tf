variable "aws_region" {
  description = "AWS region for the Week 2 GPU lab."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU EC2 instance type."
  type        = string
  default     = "g4dn.xlarge"
}

variable "vpc_cidr" {
  description = "CIDR block for the Week 2 lab VPC."
  type        = string
  default     = "10.20.20.0/24"
}

variable "subnet_cidr" {
  description = "CIDR block for the GPU subnet."
  type        = string
  default     = "10.20.20.0/26"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GiB."
  type        = number
  default     = 40
}
