variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
  default     = "ACW_AICL_Webapp"
}

variable "location" {
  description = "The location/region where the resources will be created"
  type        = string
  default     = "East US"
}

variable "vnet_name" {
  description = "The name of the virtual network"
  type        = string
  default     = "aicl_virtual_network"
}

variable "vnet_address_space" {
  description = "The address space for the virtual network"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_name" {
  description = "The name of the subnet"
  type        = string
  default     = "internal"
}

variable "subnet_prefix" {
  description = "The address prefix for the subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "public_ip_name" {
  description = "The name of the public IP address"
  type        = string
  default     = "example-pip"
}

variable "network_interface_name" {
  description = "The name of the network interface"
  type        = string
  default     = "example-nic"
}

variable "vm_name" {
  description = "The name of the virtual machine"
  type        = string
  default     = "capstone-webapp"
}

variable "admin_username" {
  description = "The admin username for the virtual machine"
  type        = string
  default     = "azureuser"
}

variable "vm_size" {
  description = "The size of the virtual machine"
  type        = string
  default     = "Standard_DS1_v2"
}

variable "ssh_public_key" {
  description = "The public SSH key to access the virtual machine"
  type        = string
}
