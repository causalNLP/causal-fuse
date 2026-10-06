import os
import json
import numpy as np
from datetime import datetime
import ast
import boto3
import time

# Configuration
DATASET_NAME = "sachs"
NUM_TRIALS = 30
MAX_TOKENS = 5000
PROMPT_FORMAT = "basic_prompt_v_2"
# Fill in with your own key.
# API_KEY = ""

# LLM models configuration
LLM_MODELS = {
    # "claude_3_haiku": {
    #     "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    #     "model_name": "Claude 3 Haiku",
    #     "region": "us-east-1"
    # },
    # "claude_3_sonnet": {
    #     "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
    #     "model_name": "Claude 3 Sonnet",
    #     "region": "us-east-1"
    # },
    # "claude_3_5_sonnet": {
    #     "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    #     "model_name": "Claude 3.5 Sonnet",
    #     "region": "us-east-1"
    # },
    # "claude_3_opus": {
    #     "model_id": "us.anthropic.claude-3-opus-20240229-v1:0",
    #     "model_name": "Claude 3 Opus",
    #     "region": "us-east-1"
    # },
    # "claude_3_5_haiku": {
    #     "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "model_name": "Claude 3.5 Haiku",
    #     "region": "us-east-1"
    # },
    # "claude_3_5_sonnet_inference": {
    #     "model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    #     "model_name": "Claude 3.5 Sonnet (inference profile)",
    #     "region": "us-east-1"
    # },
    # "claude_3_7_sonnet": {
    #     "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "model_name": "Claude 3.7 Sonnet",
    #     "region": "us-east-1"
    # },
    # "claude_opus_4": {
    #     "model_id": "us.anthropic.claude-opus-4-20250514-v1:0",
    #     "model_name": "Claude Opus 4",
    #     "region": "us-east-1"
    # },
    # "claude_sonnet_4": {
    #     "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    #     "model_name": "Claude Sonnet 4",
    #     "region": "us-east-1"
    # },
    # "claude_opus_4_1": {
    #     "model_id": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    #     "model_name": "Claude Opus 4.1",
    #     "region": "us-east-1"
    # }
    # "gpt_oss_20b": {
    #     "model_id": "openai.gpt-oss-20b-1:0",
    #     "model_name": "OpenAI GPT OSS 20B",
    #     "region": "us-west-2"
    # },
    # "gpt_oss_120b": {
    #     "model_id": "openai.gpt-oss-120b-1:0",
    #     "model_name": "OpenAI GPT OSS 120B",
    #     "region": "us-west-2"
    # },
    "llama_4_scout": {
        "model_id": "us.meta.llama4-scout-17b-instruct-v1:0",
        "model_name": "Llama 4 Scout 17B Instruct",
        "region": "us-east-1"
    },
    # "llama_4_maverick": {
    #     "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
    #     "model_name": "Llama 4 Maverick 17B Instruct",
    #     "region": "us-east-1"
    # }
}

def call_bedrock_with_retry(prompt, api_key, model_id="anthropic.claude-3-haiku-20240307-v1:0", max_tokens=200, max_retries=10, region="us-east-1"):
    """
    Call Amazon Bedrock with adaptive retry logic
    
    Args:
        prompt (str): Input prompt text
        api_key (str): Bedrock API key
        model_id (str): Bedrock model ID to use
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts
        region (str): AWS region for the client
        
    Returns:
        tuple: (response_text, wait_time_used, attempts_made)
    """
    
    # Set the API key as environment variable
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = api_key
    
    # Create an Amazon Bedrock client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region
    )
    
    wait_time = 2  # Only used for retries
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:  # Wait before retry attempts
                print(f"  Waiting {wait_time}s before attempt {attempt + 1}...")
                time.sleep(wait_time)
            
            # Create request body based on model type
            if "meta.llama" in model_id:
                # Llama models use invoke_model with specific format
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                })
                response = client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
            elif "openai.gpt-oss" in model_id:
                # GPT-OSS models use different format
                body = json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                })
                response = client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
            else:
                # Claude models use anthropic format
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "messages": messages
                })
                response = client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
            
            # Extract response text based on model type
            response_body = json.loads(response['body'].read())
            print(f"  Raw response body: {response_body}")
            
            if "meta.llama" in model_id:
                # Llama models response format
                response_text = response_body['generation']
            elif "openai.gpt-oss" in model_id:
                # GPT-OSS response format
                if 'choices' in response_body and len(response_body['choices']) > 0:
                    choice = response_body['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        full_response = choice['message']['content']
                        # Extract content after </reasoning> tag if present
                        if '</reasoning>' in full_response:
                            response_text = full_response.split('</reasoning>')[-1].strip()
                        else:
                            response_text = full_response
                    elif 'text' in choice:
                        response_text = choice['text']
                    else:
                        response_text = str(choice)
                else:
                    response_text = str(response_body)
            else:
                # Claude response format
                response_text = response_body['content'][0]['text']
            
            return response_text, wait_time, attempt + 1
            
        except Exception as e:
            print(f"  API call error: {e}")
            if "429" in str(e) or "throttl" in str(e).lower():
                wait_time += 2
                print(f"  Rate limit hit (attempt {attempt + 1}), increasing wait to {wait_time}s")
                if attempt == max_retries:
                    raise Exception(f"Failed after {max_retries + 1} attempts. Final wait time: {wait_time}s")
            else:
                raise e
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

def load_prompt(prompt_name):
    """Load prompt template from prompts.json."""
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found at {prompts_path}")
    
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    if prompt_name not in prompts:
        available = list(prompts.keys())
        raise KeyError(f"Prompt '{prompt_name}' not found. Available prompts: {available}")
    
    return prompts[prompt_name]["template"]

def load_dataset(dataset_name):
    """Load real-world dataset and return variable names."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "real-world-data", f"{dataset_name}-real-world-data.npy")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
    
    adj_matrix, data, variable_names = np.load(dataset_path, allow_pickle=True)
    return list(variable_names)

def is_valid_dag(matrix):
    """Check if matrix represents a valid DAG"""
    n = len(matrix)
    
    # Check square
    if any(len(row) != n for row in matrix):
        return False
    
    # Check binary
    for row in matrix:
        for val in row:
            if val not in [0, 1]:
                return False
    
    # Check no self-loops
    for i in range(n):
        if matrix[i][i] != 0:
            return False
    
    # Check acyclic using DFS
    def has_cycle_dfs():
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def visit(node):
            if color[node] == GRAY:
                return True  # Back edge found
            if color[node] == BLACK:
                return False
            
            color[node] = GRAY
            for neighbor in range(n):
                if matrix[node][neighbor] == 1:
                    if visit(neighbor):
                        return True
            color[node] = BLACK
            return False
        
        for i in range(n):
            if color[i] == WHITE:
                if visit(i):
                    return True
        return False
    
    return not has_cycle_dfs()

def create_prompt(variable_names):
    """Create the prompt with variable names using template."""
    var_names_str = ", ".join(variable_names)
    matrix_size = len(variable_names)
    
    # Create example matrix with correct dimensions
    example_row = "[" + ",".join(["0"] * matrix_size) + "]"
    example_matrix = "[" + ", ".join([example_row] * matrix_size) + "]"
    
    # Load prompt template
    prompt_template = load_prompt(PROMPT_FORMAT)
    
    # Format template with variables
    prompt = prompt_template.format(
        var_names_str=var_names_str,
        matrix_size=matrix_size,
        example_matrix=example_matrix
    )
    
    return prompt

def parse_adjacency_matrix(response_text, expected_size):
    """
    Parse LLM response to extract adjacency matrix.
    
    Returns:
        tuple: (success, matrix_or_error_msg)
        - If success=True: matrix_or_error_msg is numpy array
        - If success=False: matrix_or_error_msg is error message string
    """
    try:
        text = response_text.strip()
        
        # Try bracket format first [[...]]
        start_idx = text.find('[[')
        if start_idx != -1:
            bracket_count = 0
            end_idx = start_idx
            
            for i in range(start_idx, len(text)):
                if text[i] == '[':
                    bracket_count += 1
                elif text[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if bracket_count == 0:
                matrix_str = text[start_idx:end_idx]
                matrix_str = ''.join(matrix_str.split())
                matrix = ast.literal_eval(matrix_str)
            else:
                return False, "Unmatched brackets in matrix"
        else:
            # Try space-separated format
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and all(c in '01 ' for c in line):
                    row = [int(x) for x in line.split()]
                    if len(row) == expected_size:
                        lines.append(row)
            
            if len(lines) != expected_size:
                return False, f"Found {len(lines)} rows, expected {expected_size}"
            
            matrix = lines
        
        # Convert to numpy array
        adj_matrix = np.array(matrix)
        
        # Validate dimensions
        if adj_matrix.shape != (expected_size, expected_size):
            return False, f"Matrix has shape {adj_matrix.shape}, expected ({expected_size}, {expected_size})"
        
        # Validate values are 0 or 1
        if not np.all(np.isin(adj_matrix, [0, 1])):
            return False, "Matrix contains values other than 0 and 1"
        
        # Validate is a valid DAG (no cycles)
        if not is_valid_dag(adj_matrix.tolist()):
            return False, "Matrix contains cycles (not a valid DAG)"
        
        return True, adj_matrix
        
    except Exception as e:
        return False, f"Parsing error: {str(e)}"

def find_next_experiment_number():
    """Find the next experiment number."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "guess-llms-results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return 1
    
    existing_experiments = [d for d in os.listdir(results_dir) if d.startswith("guess_experiment_")]
    if not existing_experiments:
        return 1
    
    numbers = []
    for exp_dir in existing_experiments:
        try:
            num = int(exp_dir.split("_")[-1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1

def run_experiment():
    """Run the complete experiment."""
    print("Starting LLM Guess Experiment...")
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    variable_names = load_dataset(DATASET_NAME)
    print(f"Variable names: {variable_names}")
    
    # Create experiment directory
    exp_num = find_next_experiment_number()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(project_root, "guess-llms-results", f"guess_experiment_{exp_num}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    
    # Create prompt
    prompt = create_prompt(variable_names)
    
    # Save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "prompt_format": PROMPT_FORMAT,
        "num_trials": NUM_TRIALS,
        "variable_names": variable_names,
        "timestamp": datetime.now().isoformat(),
        "experiment_number": exp_num
    }
    
    with open(os.path.join(exp_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run experiments for each model
    for model_key, model_info in LLM_MODELS.items():
        print(f"\nTesting {model_info['model_name']}...")
        
        model_results = {
            "model_id": model_info["model_id"],
            "model_name": model_info["model_name"],
            "trials": []
        }
        
        for trial in range(NUM_TRIALS):
            print(f"  Trial {trial + 1}/{NUM_TRIALS}")
            
            trial_success = False
            max_retries = 2
            
            for retry in range(max_retries + 1):
                try:
                    response, wait_time, attempts = call_bedrock_with_retry(
                        prompt, 
                        API_KEY, 
                        model_info["model_id"], 
                        max_tokens=MAX_TOKENS,
                        region=model_info.get("region", "us-east-1")
                    )
                    
                    # Try to parse the adjacency matrix with detailed error reporting
                    parse_success, matrix_or_error = parse_adjacency_matrix(response, len(variable_names))
                    
                    if parse_success:
                        trial_result = {
                            "trial_number": trial + 1,
                            "response": response,
                            "adjacency_matrix": matrix_or_error.tolist(),
                            "success": True,
                            "error": None,
                            "wait_time": wait_time,
                            "attempts": attempts,
                            "parsing_retries": retry
                        }
                        
                        print(f"    ✓ Success after {attempts} attempts")
                        trial_success = True
                        break
                    else:
                        # Detailed error reporting
                        error_msg = str(matrix_or_error)
                        if "No matrix pattern found" in error_msg:
                            failure_type = "PARSING_ERROR: No matrix format detected"
                        elif "Unmatched brackets" in error_msg:
                            failure_type = "PARSING_ERROR: Malformed matrix brackets"
                        elif "Matrix has shape" in error_msg:
                            failure_type = "DIMENSION_ERROR: Wrong matrix size"
                        elif "contains values other than 0 and 1" in error_msg:
                            failure_type = "VALUE_ERROR: Non-binary values in matrix"
                        elif "contains cycles" in error_msg:
                            failure_type = "CYCLE_ERROR: Matrix is not a valid DAG"
                        elif "Parsing error" in error_msg:
                            failure_type = "PARSING_ERROR: Matrix extraction failed"
                        else:
                            failure_type = f"UNKNOWN_ERROR: {error_msg}"
                        
                        if retry < max_retries:
                            print(f"    ✗ {failure_type} - Retry {retry + 1}/{max_retries}")
                        else:
                            print(f"    ✗ FINAL FAILURE: {failure_type}")
                            trial_result = {
                                "trial_number": trial + 1,
                                "response": response,
                                "adjacency_matrix": None,
                                "success": False,
                                "error": failure_type,
                                "detailed_error": error_msg,
                                "wait_time": wait_time,
                                "attempts": attempts,
                                "parsing_retries": retry
                            }
                        
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "throttl" in error_msg.lower():
                        failure_type = "API_ERROR: Rate limit exceeded"
                    elif "timeout" in error_msg.lower():
                        failure_type = "API_ERROR: Request timeout"
                    elif "bedrock" in error_msg.lower():
                        failure_type = "API_ERROR: Bedrock service error"
                    else:
                        failure_type = f"API_ERROR: {error_msg}"
                    
                    if retry < max_retries:
                        print(f"    ✗ {failure_type} - Retry {retry + 1}/{max_retries}")
                    else:
                        print(f"    ✗ FINAL FAILURE: {failure_type}")
                        trial_result = {
                            "trial_number": trial + 1,
                            "response": None,
                            "adjacency_matrix": None,
                            "success": False,
                            "error": failure_type,
                            "detailed_error": error_msg,
                            "wait_time": None,
                            "attempts": None,
                            "parsing_retries": retry
                        }
            
            if trial_success or retry == max_retries:
                model_results["trials"].append(trial_result)
        
        # Save model results
        results_filename = f"{model_key}_results.json"
        with open(os.path.join(exp_dir, results_filename), 'w') as f:
            json.dump(model_results, f, indent=2)
        
        print(f"  Saved results to {results_filename}")
    
    print(f"\nExperiment {exp_num} completed! Results saved to: {exp_dir}")

if __name__ == "__main__":
    run_experiment()