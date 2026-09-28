# ============================================================
# DATA SOURCES
# ============================================================

data "aws_caller_identity" "current" {}

# Resolve the current AWS Deep Learning Ubuntu 22.04 GPU AMI.
# This avoids hard-coding an AMI ID.
data "aws_ssm_parameter" "gpu_dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-with-single-cuda-ubuntu-22.04/latest/ami-id"
}

# Find availability zones where the requested GPU instance type
# is offered.
data "aws_ec2_instance_type_offerings" "gpu" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }

  location_type = "availability-zone"
}

locals {
  selected_az = sort(data.aws_ec2_instance_type_offerings.gpu.locations)[0]
}

# ============================================================
# VPC
# ============================================================

resource "aws_vpc" "week02" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "ai-week02-gpu-vpc"
  }
}

# ============================================================
# INTERNET GATEWAY
# ============================================================

resource "aws_internet_gateway" "week02" {
  vpc_id = aws_vpc.week02.id

  tags = {
    Name = "ai-week02-gpu-igw"
  }
}

# ============================================================
# PUBLIC SUBNET
# ============================================================

resource "aws_subnet" "gpu" {
  vpc_id                  = aws_vpc.week02.id
  cidr_block              = var.subnet_cidr
  availability_zone       = local.selected_az
  map_public_ip_on_launch = true

  tags = {
    Name = "ai-week02-gpu-subnet"
  }
}

# ============================================================
# ROUTING
# ============================================================

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.week02.id

  tags = {
    Name = "ai-week02-public-rt"
  }
}

resource "aws_route" "internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.week02.id
}

resource "aws_route_table_association" "gpu" {
  subnet_id      = aws_subnet.gpu.id
  route_table_id = aws_route_table.public.id
}

# ============================================================
# SECURITY GROUP
# ============================================================

resource "aws_security_group" "gpu" {
  name        = "ai-week02-gpu-sg"
  description = "Week 2 GPU lab - SSM management only"
  vpc_id      = aws_vpc.week02.id

  # Intentionally no ingress rules.
  # SSH is not exposed.
  # Administration is performed through AWS Systems Manager.

  egress {
    description = "Allow outbound traffic for AWS APIs, packages and model downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ai-week02-gpu-sg"
  }
}

# ============================================================
# IAM ROLE FOR SSM
# ============================================================

resource "aws_iam_role" "gpu_ssm" {
  name = "ai-week02-gpu-ssm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "ec2.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "ai-week02-gpu-ssm-role"
  }
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.gpu_ssm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gpu" {
  name = "ai-week02-gpu-instance-profile"
  role = aws_iam_role.gpu_ssm.name
}

# ============================================================
# GPU EC2 INSTANCE
# ============================================================

resource "aws_instance" "gpu" {
  ami           = data.aws_ssm_parameter.gpu_dlami.value
  instance_type = var.instance_type

  subnet_id              = aws_subnet.gpu.id
  vpc_security_group_ids = [aws_security_group.gpu.id]

  associate_public_ip_address = true

  iam_instance_profile = aws_iam_instance_profile.gpu.name

  # Require IMDSv2.
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    encrypted             = true
    delete_on_termination = true
  }

  tags = {
    Name = "ai-week02-gpu01"
  }

  depends_on = [
    aws_route.internet,
    aws_route_table_association.gpu,
    aws_iam_role_policy_attachment.ssm_core
  ]
}
