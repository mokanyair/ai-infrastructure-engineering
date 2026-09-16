output "instance_id" {
  description = "EC2 instance ID for Systems Manager access."
  value       = aws_instance.inference.id
}

output "instance_type" {
  description = "Deployed EC2 instance type."
  value       = aws_instance.inference.instance_type
}

output "ami_id" {
  description = "Actual AMI used for the experiment."
  value       = nonsensitive(aws_instance.inference.ami)
}

output "aws_region" {
  description = "AWS Region."
  value       = var.aws_region
}

output "vpc_id" {
  description = "Lab VPC ID."
  value       = aws_vpc.lab.id
}

output "security_group_id" {
  description = "Inference security group ID."
  value       = aws_security_group.inference.id
}
