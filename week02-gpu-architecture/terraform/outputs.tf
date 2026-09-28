output "account_id" {
  description = "AWS account ID used for the deployment."
  value       = data.aws_caller_identity.current.account_id
}

output "region" {
  description = "AWS region."
  value       = var.aws_region
}

output "availability_zone" {
  description = "Availability zone selected for the GPU."
  value       = local.selected_az
}

output "instance_id" {
  description = "GPU EC2 instance ID."
  value       = aws_instance.gpu.id
}

output "instance_type" {
  description = "GPU EC2 instance type."
  value       = aws_instance.gpu.instance_type
}

output "private_ip" {
  description = "Private IP address."
  value       = aws_instance.gpu.private_ip
}

output "public_ip" {
  description = "Temporary public IP used for outbound internet connectivity."
  value       = aws_instance.gpu.public_ip
}

output "dlami_id" {
  description = "AWS Deep Learning AMI resolved from SSM."
  value       = nonsensitive(data.aws_ssm_parameter.gpu_dlami.value)
}

output "ssm_command" {
  description = "Command used to connect through AWS Systems Manager."
  value       = "aws ssm start-session --target ${aws_instance.gpu.id} --region ${var.aws_region}"
}

output "stop_command" {
  description = "Emergency cost-control command."
  value       = "aws ec2 stop-instances --instance-ids ${aws_instance.gpu.id} --region ${var.aws_region}"
}

output "destroy_reminder" {
  description = "Destroy the lab when testing is complete."
  value       = "terraform destroy"
}
