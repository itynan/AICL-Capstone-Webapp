provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "my_resource_group" {
  name     = "ACW_AICL_Webapp"
  location = "East US"
}

resource "azurerm_virtual_network" "aicl_virtual_network" {
  name                = "aicl_virtual_network"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.my_resource_group.location
  resource_group_name = azurerm_resource_group.my_resource_group.name
}

resource "azurerm_subnet" "my_subnet" {
  name                 = "internal"
  resource_group_name  = azurerm_resource_group.my_resource_group.name
  virtual_network_name = azurerm_virtual_network.aicl_virtual_network.name
  address_prefixes     = ["10.0.2.0/24"]
}

resource "azurerm_network_interface" "my_network_interface" {
  name                = "example-nic"
  location            = azurerm_resource_group.my_resource_group.location
  resource_group_name = azurerm_resource_group.my_resource_group.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.my_subnet.id
    private_ip_address_allocation = "Static"
    private_ip_address            = "10.0.2.4"  # Choose an appropriate IP from your subnet range
    public_ip_address_id          = azurerm_public_ip.my_public_ip.id
  }
}

resource "azurerm_public_ip" "my_public_ip" {
  name                = "example-pip"
  location            = azurerm_resource_group.my_resource_group.location
  resource_group_name = azurerm_resource_group.my_resource_group.name
  allocation_method   = "Dynamic"
}

resource "azurerm_network_security_group" "my_security_group" {
  name                = "example-nsg"
  location            = azurerm_resource_group.my_resource_group.location
  resource_group_name = azurerm_resource_group.my_resource_group.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_interface_security_group_association" "my_net_interface_sec_gp_assoc" {
  network_interface_id      = azurerm_network_interface.my_network_interface.id
  network_security_group_id = azurerm_network_security_group.my_security_group.id
}
resource "azurerm_linux_virtual_machine" "my_linux_virtual_machine" {
  name                = "capstone-webapp"
  resource_group_name = azurerm_resource_group.my_resource_group.name
  location            = azurerm_resource_group.my_resource_group.location
  size                = "Standard_DS1_v2"

  admin_username = "azureuser"
  disable_password_authentication = true

  network_interface_ids = [
    azurerm_network_interface.my_network_interface.id,
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "UbuntuServer"
    sku       = "18_04-lts-gen2"
    version   = "latest"
  }

  computer_name  = "hostname"

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }
}

output "public_ip_address" {
  value = azurerm_public_ip.my_public_ip.ip_address
}
