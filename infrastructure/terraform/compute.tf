data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  selected_ami = (
    var.ami_id != null
    ? var.ami_id
    : data.aws_ssm_parameter.al2023.value
  )
}

resource "aws_instance" "inference" {
  ami                         = local.selected_ami
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.inference.id]
  iam_instance_profile        = aws_iam_instance_profile.inference.name
  associate_public_ip_address = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "ai-infra-week01-root"
    }
  }

  monitoring = false

  tags = {
    Name = "ai-infra-week01-inference"
  }

  depends_on = [
    aws_iam_role_policy_attachment.ssm,
    aws_route_table_association.public
  ]
}
