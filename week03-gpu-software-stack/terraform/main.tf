data "aws_ec2_instance_type_offerings" "g5" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }

  location_type = "availability-zone"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_vpc" "week03" {
  cidr_block           = "10.20.30.0/24"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name    = "${var.project_name}-vpc"
    Project = "AI-Infrastructure-Week03"
  }
}

resource "aws_internet_gateway" "week03" {
  vpc_id = aws_vpc.week03.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "week03" {
  vpc_id                  = aws_vpc.week03.id
  cidr_block              = "10.20.30.0/26"
  availability_zone       = sort(data.aws_ec2_instance_type_offerings.g5.locations)[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public"
  }
}

resource "aws_route_table" "week03" {
  vpc_id = aws_vpc.week03.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.week03.id
  }

  tags = {
    Name = "${var.project_name}-rt"
  }
}

resource "aws_route_table_association" "week03" {
  subnet_id      = aws_subnet.week03.id
  route_table_id = aws_route_table.week03.id
}

resource "aws_security_group" "week03" {
  name        = "${var.project_name}-sg"
  description = "Week 3 GPU lab - outbound only"
  vpc_id      = aws_vpc.week03.id

  egress {
    description = "Allow outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

resource "aws_iam_role" "ssm" {
  name = "${var.project_name}-ssm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ssm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ssm" {
  name = "${var.project_name}-profile"
  role = aws_iam_role.ssm.name
}

resource "aws_instance" "gpu" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.week03.id
  vpc_security_group_ids      = [aws_security_group.week03.id]
  iam_instance_profile        = aws_iam_instance_profile.ssm.name
  associate_public_ip_address = true

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name    = "${var.project_name}-gpu01"
    Project = "AI-Infrastructure-Week03"
    Week    = "03"
  }
}
