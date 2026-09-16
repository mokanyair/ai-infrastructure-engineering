resource "aws_security_group" "inference" {
  name_prefix = "ai-infra-week01-"
  description = "Security group for CPU inference lab"
  vpc_id      = aws_vpc.lab.id

  tags = {
    Name = "ai-infra-week01-sg"
  }
}

resource "aws_vpc_security_group_egress_rule" "https" {
  security_group_id = aws_security_group.inference.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "tcp"
  from_port   = 443
  to_port     = 443

  description = "Outbound HTTPS for SSM, packages and model downloads"
}
