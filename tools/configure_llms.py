import os
import sys
import json
import getpass
from pathlib import Path
import configparser
from typing import Dict, Optional

class LLMConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / '.llm_config'
        self.config_file = self.config_dir / 'config.ini'
        self.providers = ['aws_bedrock', 'openai']
        self.ensure_config_directory()
        
    def ensure_config_directory(self):
        """Create configuration directory if it doesn't exist"""
        self.config_dir.mkdir(exist_ok=True)
        
    def load_current_config(self) -> configparser.ConfigParser:
        """Load existing configuration or create new one"""
        config = configparser.ConfigParser()
        if self.config_file.exists():
            config.read(self.config_file)
        return config
    
    def save_config(self, config: configparser.ConfigParser):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            config.write(f)
            
    def set_environment_variables(self, config: configparser.ConfigParser):
        """Set environment variables based on configuration"""
        if 'aws_bedrock' in config:
            os.environ['BEDROCK_API_KEY'] = config['aws_bedrock']['bedrock_api_key']
            os.environ['AWS_DEFAULT_REGION'] = config['aws_bedrock']['aws_region']
            os.environ['AWS_ACCESS_KEY_ID'] = config['aws_bedrock']['aws_access_key_id']
            os.environ['AWS_SECRET_ACCESS_KEY'] = config['aws_bedrock']['aws_secret_access_key']
            
        if 'openai' in config:
            os.environ['OPENAI_API_KEY'] = config['openai']['api_key']
            
    def configure_aws_bedrock(self) -> Dict[str, str]:
        """Interactive configuration for AWS Bedrock"""
        print("\n=== AWS Bedrock Configuration ===")
        config = {
            'bedrock_api_key': getpass.getpass('Enter Bedrock API Key: '),
            'aws_region': input('Enter AWS Region (e.g., us-east-1): '),
            'aws_access_key_id': input('Enter AWS Access Key ID: '),
            'aws_secret_access_key': getpass.getpass('Enter AWS Secret Access Key: ')
        }
        return config
    
    def configure_openai(self) -> Dict[str, str]:
        """Interactive configuration for OpenAI"""
        print("\n=== OpenAI Configuration ===")
        config = {
            'api_key': getpass.getpass('Enter OpenAI API Key (starts with sk-): ')
        }
        return config
    
    def validate_aws_config(self, config: Dict[str, str]) -> bool:
        """Basic validation for AWS configuration"""
        if not all(config.values()):
            return False
        if not config['aws_region'].startswith('us-') and not config['aws_region'].startswith('eu-'):
            return False
        if not config['bedrock_api_key'].strip():
            return False
        return True
    
    def validate_openai_config(self, config: Dict[str, str]) -> bool:
        """Basic validation for OpenAI configuration"""
        return bool(config['api_key'] and config['api_key'].startswith('sk-'))
    
    def run_interactive_setup(self):
        """Run the interactive configuration process"""
        print("Welcome to the LLM Configuration Tool!")
        print("\nAvailable LLM Providers:")
        for idx, provider in enumerate(self.providers, 1):
            print(f"{idx}. {provider}")
            
        while True:
            try:
                choice = int(input("\nSelect provider (1-2): "))
                if choice not in [1, 2]:
                    raise ValueError
                break
            except ValueError:
                print("Please enter a valid number (1-2)")
                
        config = self.load_current_config()
        
        if choice == 1:  # AWS Bedrock
            aws_config = self.configure_aws_bedrock()
            if not self.validate_aws_config(aws_config):
                print("Error: Invalid AWS configuration. Please check your inputs.")
                return
                
            if 'aws_bedrock' not in config:
                config.add_section('aws_bedrock')
            for key, value in aws_config.items():
                config['aws_bedrock'][key] = value
                
        else:  # OpenAI
            openai_config = self.configure_openai()
            if not self.validate_openai_config(openai_config):
                print("Error: Invalid OpenAI API key format. Key should start with 'sk-'")
                return
                
            if 'openai' not in config:
                config.add_section('openai')
            for key, value in openai_config.items():
                config['openai'][key] = value
                
        self.save_config(config)
        self.set_environment_variables(config)
        print("\nConfiguration saved successfully!")
        print(f"Configuration file location: {self.config_file}")
        
    def show_current_config(self):
        """Display current configuration (with masked sensitive data)"""
        if not self.config_file.exists():
            print("No configuration file found.")
            return
            
        config = self.load_current_config()
        print("\nCurrent Configuration:")
        print("=====================")
        
        for section in config.sections():
            print(f"\n[{section}]")
            for key, value in config[section].items():
                if 'key' in key.lower() or 'secret' in key.lower():
                    masked_value = value[:4] + '****' + value[-4:]
                    print(f"{key}: {masked_value}")
                else:
                    print(f"{key}: {value}")

def main():
    config_manager = LLMConfigManager()
    
    while True:
        print("\nLLM Configuration Tool")
        print("1. Configure LLM Provider")
        print("2. Show Current Configuration")
        print("3. Exit")
        
        try:
            choice = int(input("\nSelect an option (1-3): "))
            if choice == 1:
                config_manager.run_interactive_setup()
            elif choice == 2:
                config_manager.show_current_config()
            elif choice == 3:
                print("Goodbye!")
                break
            else:
                print("Please enter a valid option (1-3)")
        except ValueError:
            print("Please enter a valid number")

if __name__ == "__main__":
    main()
