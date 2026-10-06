output "instance_id" {
  value = aws_instance.gpu.id
}

output "instance_type" {
  value = aws_instance.gpu.instance_type
}

output "ami_id" {
  value = data.aws_ami.ubuntu.id
}

output "availability_zone" {
  value = aws_instance.gpu.availability_zone
}

output "public_ip" {
  value = aws_instance.gpu.public_ip
}
