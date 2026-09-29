terraform {
  required_version = ">= 1.5.0"
  required_providers {
    linode = {
      source  = "linode/linode"
      version = "~> 2.13"
    }
  }
}
variable "linode_token" {
  type      = string
  sensitive = true
}
variable "region" {
  type    = string
  default = "in-maa"
}
variable "ssh_cidr" {
  type        = string
  description = "Operator's current IPv4 address followed by /32. Never use 0.0.0.0/0."
  validation {
    condition     = can(cidrhost(var.ssh_cidr, 0)) && endswith(var.ssh_cidr, "/32") && !strcontains(var.ssh_cidr, ":")
    error_message = "Supply one operator IPv4 /32."
  }
}
provider "linode" {
  token = var.linode_token
}
locals {
  secrets    = "${path.module}/../../.secrets/linode"
  public_key = trimspace(file("${local.secrets}/id_ed25519.pub"))
  cloud_config = {
    package_update = true
    packages       = ["docker.io", "docker-compose-v2", "unattended-upgrades"]
    ssh_pwauth     = false
    disable_root   = true
    ssh_keys = {
      ed25519_private = file("${local.secrets}/host_ed25519")
      ed25519_public  = trimspace(file("${local.secrets}/host_ed25519.pub"))
    }
    users = ["default", {
      name                = "deploy"
      shell               = "/bin/bash"
      lock_passwd         = true
      sudo                = ["ALL=(ALL) NOPASSWD:ALL"]
      ssh_authorized_keys = [local.public_key]
    }]
    write_files = [{
      path        = "/etc/ssh/sshd_config.d/00-sangama.conf"
      permissions = "0644"
      content     = "PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin no\nAllowUsers deploy\n"
    }]
    runcmd = [
      ["systemctl", "enable", "--now", "docker"],
      ["systemctl", "restart", "ssh"],
      ["mkdir", "-p", "/srv/sangama"],
      ["chown", "deploy:deploy", "/srv/sangama"]
    ]
  }
}
variable "mesh_relay" {
  description = "Open TCP 9000 for the admitted-mesh relay (Circuit Relay v2) on this server"
  type        = bool
  default     = false
}
resource "linode_firewall" "portal" {
  label           = "sangama-portal"
  inbound_policy  = "DROP"
  outbound_policy = "ACCEPT"
  inbound {
    label    = "operator-ssh"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "22"
    ipv4     = [var.ssh_cidr]
  }
  inbound {
    label    = "public-https"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "80,443"
    ipv4     = ["0.0.0.0/0"]
    ipv6     = ["::/0"]
  }
  dynamic "inbound" {
    for_each = var.mesh_relay ? [1] : []
    content {
      label    = "mesh-relay"
      action   = "ACCEPT"
      protocol = "TCP"
      ports    = "9000"
      ipv4     = ["0.0.0.0/0"]
    }
  }
}
resource "linode_instance" "portal" {
  label           = "sangama-portal"
  region          = var.region
  type            = "g6-standard-1"
  image           = "linode/ubuntu24.04"
  backups_enabled = true
  firewall_id     = linode_firewall.portal.id
  authorized_keys = [local.public_key]
  root_pass       = sensitive(trimspace(file("${local.secrets}/root-password")))
  metadata {
    user_data = sensitive(base64encode("#cloud-config\n${jsonencode(local.cloud_config)}"))
  }
  tags = ["sangama"]
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [metadata, root_pass]
  }
}
output "server_ip" { value = one(linode_instance.portal.ipv4) }
output "server_id" { value = linode_instance.portal.id }
