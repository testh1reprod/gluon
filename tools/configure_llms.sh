#!/bin/bash

# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced. Please run:"
    echo "source ${0}"
    exit 1
fi

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration file path
CONFIG_FILE="$HOME/.llm_config"

# Function to print colored messages
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section header
print_header() {
    local message=$1
    echo
    print_color "$BLUE" "=== $message ==="
    echo
}

# Function to validate AWS region
validate_aws_region() {
    local region=$1
    # List of valid AWS regions
    local valid_regions=("us-east-1" "us-east-2" "us-west-1" "us-west-2" "eu-west-1" "eu-central-1" "ap-southeast-1" "ap-southeast-2" "ap-northeast-1")
    
    for valid_region in "${valid_regions[@]}"; do
        if [ "$region" == "$valid_region" ]; then
            return 0
        fi
    done
    return 1
}

# Function to validate API keys
validate_api_key() {
    local key=$1
    local type=$2
    
    case $type in
        "bedrock")
            [[ $key =~ ^[0-9]{4}[0-9.]+$ ]] && return 0 ;;
        "openai")
            [[ $key =~ ^sk-[A-Za-z0-9]+$ ]] && return 0 ;;
        *)
            return 1 ;;
    esac
    return 1
}

# Function to read existing configuration into temporary variables
read_existing_config() {
    # Declare temporary variables
    declare -g tmp_BEDROCK_API_KEY=""
    declare -g tmp_AWS_DEFAULT_REGION=""
    declare -g tmp_AWS_ACCESS_KEY_ID=""
    declare -g tmp_AWS_SECRET_ACCESS_KEY=""
    declare -g tmp_OPENAI_API_KEY=""
    
    if [ -f "$CONFIG_FILE" ]; then
        while IFS='=' read -r key value; do
            if [ -n "$key" ] && [ -n "$value" ]; then
                case "$key" in
                    "BEDROCK_API_KEY") tmp_BEDROCK_API_KEY="$value" ;;
                    "AWS_DEFAULT_REGION") tmp_AWS_DEFAULT_REGION="$value" ;;
                    "AWS_ACCESS_KEY_ID") tmp_AWS_ACCESS_KEY_ID="$value" ;;
                    "AWS_SECRET_ACCESS_KEY") tmp_AWS_SECRET_ACCESS_KEY="$value" ;;
                    "OPENAI_API_KEY") tmp_OPENAI_API_KEY="$value" ;;
                esac
            fi
        done < "$CONFIG_FILE"
    fi
}

# Function to save configuration and export variables
save_configuration() {
    local provider=$1
    
    # Create or truncate the config file
    > "$CONFIG_FILE"
    
    if [ "$provider" == "bedrock" ]; then
        # Update Bedrock variables
        tmp_BEDROCK_API_KEY="$BEDROCK_API_KEY"
        tmp_AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION"
        tmp_AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
        tmp_AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
    else
        # Update OpenAI variable
        tmp_OPENAI_API_KEY="$OPENAI_API_KEY"
    fi
    
    # Save all configurations
    if [ -n "$tmp_BEDROCK_API_KEY" ]; then
        cat << EOF >> "$CONFIG_FILE"
BEDROCK_API_KEY=$tmp_BEDROCK_API_KEY
AWS_DEFAULT_REGION=$tmp_AWS_DEFAULT_REGION
AWS_ACCESS_KEY_ID=$tmp_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$tmp_AWS_SECRET_ACCESS_KEY
EOF
    fi
    
    if [ -n "$tmp_OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY=$tmp_OPENAI_API_KEY" >> "$CONFIG_FILE"
    fi
    
    # Export all variables
    if [ -n "$tmp_BEDROCK_API_KEY" ]; then
        export BEDROCK_API_KEY="$tmp_BEDROCK_API_KEY"
        export AWS_DEFAULT_REGION="$tmp_AWS_DEFAULT_REGION"
        export AWS_ACCESS_KEY_ID="$tmp_AWS_ACCESS_KEY_ID"
        export AWS_SECRET_ACCESS_KEY="$tmp_AWS_SECRET_ACCESS_KEY"
    fi
    
    if [ -n "$tmp_OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY="$tmp_OPENAI_API_KEY"
    fi
    
    # Set proper permissions
    chmod 600 "$CONFIG_FILE"
    
    print_color "$GREEN" "Configuration saved to $CONFIG_FILE"
    print_color "$GREEN" "Variables have been exported in the current session"
}

# Function to check and backup existing configuration
check_existing_config() {
    if [ -f "$CONFIG_FILE" ]; then
        print_color "$BLUE" "Existing configuration found. Creating backup..."
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
        print_color "$GREEN" "Backup created at ${CONFIG_FILE}.backup"
    fi
}

# Function to display current configuration file
display_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_color "$YELLOW" "No configuration file found at $CONFIG_FILE"
        return
    fi
    
    read_existing_config
    
    print_header "Current Configuration File"
    
    print_color "$GREEN" "AWS Bedrock Configuration:"
    if [ -n "$tmp_BEDROCK_API_KEY" ]; then
        echo "BEDROCK_API_KEY=${tmp_BEDROCK_API_KEY}"
        echo "AWS_DEFAULT_REGION=${tmp_AWS_DEFAULT_REGION}"
        echo "AWS_ACCESS_KEY_ID=${tmp_AWS_ACCESS_KEY_ID}"
        echo "AWS_SECRET_ACCESS_KEY=********"
    else
        print_color "$YELLOW" "Bedrock is not configured"
    fi
    
    echo
    print_color "$GREEN" "OpenAI Configuration:"
    if [ -n "$tmp_OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY=${tmp_OPENAI_API_KEY}"
    else
        print_color "$YELLOW" "OpenAI is not configured"
    fi
    echo
}

# Function to display current environment variables
display_env_vars() {
    print_header "Current Environment Variables"
    
    print_color "$GREEN" "AWS Bedrock Environment Variables:"
    echo "BEDROCK_API_KEY=${BEDROCK_API_KEY:-(not set)}"
    echo "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-(not set)}"
    echo "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-(not set)}"
    if [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "AWS_SECRET_ACCESS_KEY=********"
    else
        echo "AWS_SECRET_ACCESS_KEY=(not set)"
    fi
    
    echo
    print_color "$GREEN" "OpenAI Environment Variables:"
    echo "OPENAI_API_KEY=${OPENAI_API_KEY:-(not set)}"
    echo

    # Compare configuration with environment variables
    if [ -f "$CONFIG_FILE" ]; then
        read_existing_config
        
        local has_mismatch=false
        
        if [ -n "$tmp_BEDROCK_API_KEY" ] && [ "$tmp_BEDROCK_API_KEY" != "$BEDROCK_API_KEY" ]; then
            has_mismatch=true
        fi
        if [ -n "$tmp_AWS_DEFAULT_REGION" ] && [ "$tmp_AWS_DEFAULT_REGION" != "$AWS_DEFAULT_REGION" ]; then
            has_mismatch=true
        fi
        if [ -n "$tmp_AWS_ACCESS_KEY_ID" ] && [ "$tmp_AWS_ACCESS_KEY_ID" != "$AWS_ACCESS_KEY_ID" ]; then
            has_mismatch=true
        fi
        if [ -n "$tmp_AWS_SECRET_ACCESS_KEY" ] && [ "$tmp_AWS_SECRET_ACCESS_KEY" != "$AWS_SECRET_ACCESS_KEY" ]; then
            has_mismatch=true
        fi
        if [ -n "$tmp_OPENAI_API_KEY" ] && [ "$tmp_OPENAI_API_KEY" != "$OPENAI_API_KEY" ]; then
            has_mismatch=true
        fi
        
        if [ "$has_mismatch" = true ]; then
            print_color "$YELLOW" "Warning: Some environment variables don't match the configuration file."
            print_color "$YELLOW" "Run 'source $CONFIG_FILE' to sync them."
        fi
    fi
}

# Function to configure providers
configure_provider() {
    print_color "$GREEN" "Select your LLM provider to configure:"
    echo "1) AWS Bedrock"
    echo "2) OpenAI"
    read -p "Enter your choice (1/2): " provider_choice
    
    case $provider_choice in
        1)
            print_color "$BLUE" "\nConfiguring AWS Bedrock..."
            
            while true; do
                read -p "Enter your Bedrock API Key (starts with 4509): " BEDROCK_API_KEY
                if validate_api_key "$BEDROCK_API_KEY" "bedrock"; then
                    break
                fi
                print_color "$RED" "Invalid Bedrock API Key format. Please try again."
            done
            
            while true; do
                read -p "Enter your AWS region (e.g., us-east-1): " AWS_DEFAULT_REGION
                if validate_aws_region "$AWS_DEFAULT_REGION"; then
                    break
                fi
                print_color "$RED" "Invalid AWS region. Please enter a valid region."
            done
            
            read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
            read -s -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
            echo
            
            save_configuration "bedrock"
            ;;
            
        2)
            print_color "$BLUE" "\nConfiguring OpenAI..."
            
            while true; do
                read -p "Enter your OpenAI API Key (starts with sk-): " OPENAI_API_KEY
                if validate_api_key "$OPENAI_API_KEY" "openai"; then
                    break
                fi
                print_color "$RED" "Invalid OpenAI API Key format. Please try again."
            done
            
            print_color "$RED" "Note: Free-tier OpenAI accounts may be subject to rate limits."
            print_color "$RED" "We recommend using a paid OpenAI API key for seamless functionality."
            
            save_configuration "openai"
            ;;
            
        *)
            print_color "$RED" "Invalid choice. Exiting."
            return 1
            ;;
    esac
    
    print_color "$GREEN" "\nConfiguration complete!"
    display_config
}

# Main script
clear
print_color "$BLUE" "=== LLM Configuration Tool ==="
echo
print_color "$GREEN" "Select an option:"
echo "1) Configure LLM providers"
echo "2) View current configuration file"
echo "3) View current environment variables"
read -p "Enter your choice (1/2/3): " main_choice

case $main_choice in
    1)
        check_existing_config
        read_existing_config
        configure_provider
        ;;
    2)
        display_config
        ;;
    3)
        display_env_vars
        ;;
    *)
        print_color "$RED" "Invalid choice. Exiting."
        return 1
        ;;
esac
