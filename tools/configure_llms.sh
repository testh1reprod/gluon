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

# Initialize temporary variables
tmp_AWS_DEFAULT_REGION=""
tmp_AWS_ACCESS_KEY_ID=""
tmp_AWS_SECRET_ACCESS_KEY=""
tmp_OPENAI_API_KEY=""

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
    # TODO: Raise warning if bedrock is not supported or is gated in region
    local valid_regions=("us-east-1" "us-east-2" "us-west-1" "us-west-2" "eu-west-1" "eu-central-1" "ap-southeast-1" "ap-southeast-2" "ap-northeast-1")
    
    for valid_region in "${valid_regions[@]}"; do
        if [ "$region" = "$valid_region" ]; then
            return 0
        fi
    done
    return 1
}

# Function to validate API keys
validate_openai_api_key() {
    local key=$1
    [[ $key =~ ^sk-[A-Za-z0-9_-]+$ ]] && return 0
    return 1
}

# Function to read existing configuration into temporary variables
read_existing_config() {
    if [ -f "$CONFIG_FILE" ]; then
        while IFS='=' read -r key value; do
            if [ -n "$key" ] && [ -n "$value" ]; then
                case "$key" in
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
    echo "" > "$CONFIG_FILE" || { print_color "$RED" "Error: Cannot write to '$CONFIG_FILE'"; return 1; }
    
    if [ "$provider" = "bedrock" ]; then
        # Update AWS variables
        tmp_AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION"
        tmp_AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
        tmp_AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
    else
        # Update OpenAI variable
        tmp_OPENAI_API_KEY="$OPENAI_API_KEY"
    fi
    
    # Save all configurations
    if [ -n "$tmp_AWS_ACCESS_KEY_ID" ]; then
        echo "AWS_DEFAULT_REGION=$tmp_AWS_DEFAULT_REGION" >> "$CONFIG_FILE"
        echo "AWS_ACCESS_KEY_ID=$tmp_AWS_ACCESS_KEY_ID" >> "$CONFIG_FILE"
        echo "AWS_SECRET_ACCESS_KEY=$tmp_AWS_SECRET_ACCESS_KEY" >> "$CONFIG_FILE"
    fi
    
    if [ -n "$tmp_OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY=$tmp_OPENAI_API_KEY" >> "$CONFIG_FILE"
    fi
    
    # Export all variables
    if [ -n "$tmp_AWS_ACCESS_KEY_ID" ]; then
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
    if [ -n "$tmp_AWS_ACCESS_KEY_ID" ]; then
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
    echo -n "Enter your choice (1/2): "
    read provider_choice
    
    case $provider_choice in
        1)
            print_color "$BLUE" "\nConfiguring AWS Bedrock..."
            
            while true; do
                echo -n "Enter your AWS region (e.g., us-east-1): "
                read AWS_DEFAULT_REGION
                if validate_aws_region "$AWS_DEFAULT_REGION"; then
                    break
                fi
                print_color "$RED" "Invalid AWS region. Please enter a valid region."
            done
            
            echo -n "Enter your AWS Access Key ID: "
            read AWS_ACCESS_KEY_ID
            echo -n "Enter your AWS Secret Access Key: "
            read -s AWS_SECRET_ACCESS_KEY
            echo
            
            save_configuration "bedrock"
            ;;
            
        2)
            print_color "$BLUE" "\nConfiguring OpenAI..."
            
            while true; do
                echo -n "Enter your OpenAI API Key (starts with sk-): "
                read OPENAI_API_KEY
                if validate_openai_api_key "$OPENAI_API_KEY"; then
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
print_color "$BLUE" "=== AGA LLM Configuration Tool ==="
echo
print_color "$GREEN" "Select an option:"
echo "1) Configure LLM providers"
echo "2) View current configuration file"
echo "3) View current environment variables"
echo -n "Enter your choice (1/2/3): "
read main_choice

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
