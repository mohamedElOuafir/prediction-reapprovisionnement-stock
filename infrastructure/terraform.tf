terraform {
  required_providers {
    aws = {
        source = "hashicorp/aws"
        version = "~> 6.0"
    }
  }

  required_version = ">= 1.2"

  backend "s3" {
    bucket = "terraform-state-bucket-602340147355-eu-west-3-an"
    key    = "reapprovisionnement/terraform.tfstate"
    region = "eu-west-3"
  }
}
