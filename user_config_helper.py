#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from os import linesep, path
from sys import argv
from typing import Optional
import helper

# This should not change unless the Speech REST API changes.
PARTIAL_SPEECH_ENDPOINT = ".api.cognitive.microsoft.com"


def load_env_file(env_path: str = ".env") -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    
    # Check if .env file exists
    if not path.exists(env_path):
        return env_vars
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
    except Exception as e:
        print(f"Warning: Failed to load .env file: {e}")
    
    return env_vars


def get_cmd_option(option: str) -> Optional[str]:
    argc = len(argv)
    if option.lower() in list(map(lambda arg: arg.lower(), argv)):
        index = argv.index(option)
        if index < argc - 1:
            # We found the option (for example, "--output"), so advance from that to the value (for example, "filename").
            return argv[index + 1]
        else:
            return None
    else:
        return None


def cmd_option_exists(option: str) -> bool:
    return option.lower() in list(map(lambda arg: arg.lower(), argv))


def user_config_from_args(usage: str) -> helper.Read_Only_Dict:
    # Load environment variables from .env file
    env_vars = load_env_file()
    
    input_audio_url = get_cmd_option("--input")
    input_file_path = get_cmd_option("--jsonInput")
    
    # Get container URL from command line (priority) or .env file (fallback)
    container_url = get_cmd_option("--containerUrl")
    if container_url is None:
        container_url = env_vars.get("CONTAINER_URL")
    
    if input_audio_url is None and input_file_path is None and container_url is None:
        raise RuntimeError(f"Please specify either --input, --jsonInput, or --containerUrl.{linesep}{usage}")

    # Get speech key from command line (priority) or .env file (fallback)
    speech_subscription_key = get_cmd_option("--speechKey")
    if speech_subscription_key is None:
        speech_subscription_key = env_vars.get("SPEECH_KEY")
    if speech_subscription_key is None and input_file_path is None:
        err_msg = "Missing Speech subscription key. Provide via --speechKey or SPEECH_KEY in .env file."
        raise RuntimeError(f"{err_msg}{linesep}{usage}")
    
    # Get speech region from command line (priority) or .env file (fallback)
    speech_region = get_cmd_option("--speechRegion")
    if speech_region is None:
        speech_region = env_vars.get("SPEECH_REGION")
    if speech_region is None and input_file_path is None:
        raise RuntimeError(f"Missing Speech region. Provide via --speechRegion or SPEECH_REGION in .env file.{linesep}{usage}")

    # Text analytics is controlled by --useTextAnalytics flag
    # If flag is present, use language key/endpoint from command line or .env
    # If flag is absent, text analytics is disabled even if credentials exist in .env
    use_text_analytics = cmd_option_exists("--useTextAnalytics")
    
    language_subscription_key = None
    language_endpoint = None
    
    if use_text_analytics:
        # Get language key from command line (priority) or .env file (fallback)
        language_subscription_key = get_cmd_option("--languageKey")
        if language_subscription_key is None:
            language_subscription_key = env_vars.get("LANGUAGE_KEY")
        
        # Get language endpoint from command line (priority) or .env file (fallback)
        language_endpoint = get_cmd_option("--languageEndpoint")
        if language_endpoint is None:
            language_endpoint = env_vars.get("LANGUAGE_ENDPOINT")
        
        if language_endpoint is not None:
            language_endpoint = language_endpoint.replace("https://", "")
        
        # Validate that both key and endpoint are provided if analytics is enabled
        if language_subscription_key is None or language_endpoint is None:
            print("Warning: --useTextAnalytics specified but missing credentials.")
            print("  Language Key or Endpoint not found in command line or .env file.")
            print("  Conversation analytics will be disabled.")
            language_subscription_key = None
            language_endpoint = None

    language = get_cmd_option("--language")
    if language is None:
        language = "en"
    locale = get_cmd_option("--locale")
    if locale is None:
        locale = "en-US"
    
    # Parse candidate locales for multi-language support from command line or .env
    candidate_locales_str = get_cmd_option("--candidateLocales")
    if candidate_locales_str is None:
        candidate_locales_str = env_vars.get("CANDIDATE_LOCALES")
    
    candidate_locales = None
    if candidate_locales_str:
        candidate_locales = [loc.strip() for loc in candidate_locales_str.split(",")]
    
    language_identification_mode = get_cmd_option("--languageIdMode")
    if language_identification_mode is None:
        language_identification_mode = "Continuous"
    
    # Get Whisper model ID if specified
    whisper_model = get_cmd_option("--whisperModel")
    
    # Get output folder for batch processing from command line or .env
    output_folder = get_cmd_option("--outputFolder")
    if output_folder is None:
        output_folder = env_vars.get("OUTPUT_FOLDER")
    if output_folder is None and container_url is not None:
        output_folder = "output"
    
    # Get output container URL for writing results to blob storage from command line or .env
    output_container_url = get_cmd_option("--outputContainerUrl")
    if output_container_url is None:
        output_container_url = env_vars.get("OUTPUT_CONTAINER_URL")

    return helper.Read_Only_Dict({
        "use_stereo_audio": cmd_option_exists("--stereo"),
        "language": language,
        "locale": locale,
        "candidate_locales": candidate_locales,
        "language_identification_mode": language_identification_mode,
        "input_audio_url": input_audio_url,
        "input_file_path": input_file_path,
        "output_file_path": get_cmd_option("--output"),
        "speech_subscription_key": speech_subscription_key,
        "speech_endpoint": f"{speech_region}{PARTIAL_SPEECH_ENDPOINT}",
        "language_subscription_key": language_subscription_key,
        "language_endpoint": language_endpoint,
        "whisper_model": whisper_model,
        "container_url": container_url,
        "output_folder": output_folder,
        "output_container_url": output_container_url,
    })
