variable "aws_region" {
  description = "AWS region for Week 3 GPU lab"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "Week 3 GPU instance"
  type        = string
  default     = "g5.xlarge"
}

variable "root_volume_size" {
  description = "Root gp3 volume size"
  type        = number
  default     = 120
}

variable "project_name" {
  type    = string
  default = "ai-infra-week03"
}
