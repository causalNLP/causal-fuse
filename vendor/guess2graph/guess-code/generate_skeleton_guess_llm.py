import os
import json
import numpy as np
from datetime import datetime
import ast
import boto3
import time
import random
import re

# Configuration
DATASET_NAME = "sachs"
NUM_TRIALS = 20
MAX_TOKENS = 5000
PROMPT_FORMAT = "skeleton_prompt_v_4"
# TODO: Replace with your AWS Bedrock API key
# You can obtain this from your AWS account with Bedrock access
API_KEY = "YOUR_AWS_BEDROCK_API_KEY_HERE"

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
    "claude_opus_4_1": {
        "model_id": "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "model_name": "Claude Opus 4.1",
        "region": "us-east-1"
    }
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
    # "llama_4_scout": {
    #     "model_id": "us.meta.llama4-scout-17b-instruct-v1:0",
    #     "model_name": "Llama 4 Scout 17B Instruct",
    #     "region": "us-east-1"
    # },
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

def get_causal_ordering(adj_matrix):
    """Extract causal ordering from ground truth DAG using topological sort."""
    n = len(adj_matrix)
    in_degree = [0] * n
    
    # Calculate in-degrees
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1
    
    # Topological sort
    queue = [i for i in range(n) if in_degree[i] == 0]
    ordering = []
    
    while queue:
        node = queue.pop(0)
        ordering.append(node)
        
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    return ordering

def parse_edge_pairs(response_text, variable_names):
    """Parse LLM response to extract edge pairs."""
    try:
        text = response_text.strip()
        
        # Find EDGES: tag
        edges_start = text.find('EDGES:')
        if edges_start == -1:
            return False, "No EDGES: tag found in response"
        
        edges_text = text[edges_start + 6:].strip()
        
        # Extract pairs using regex
        pair_pattern = r'\(([^,]+),\s*([^)]+)\)'
        matches = re.findall(pair_pattern, edges_text)
        
        if not matches:
            return False, "No valid edge pairs found"
        
        # Validate variable names and create unique pairs
        valid_pairs = set()
        for var1, var2 in matches:
            var1, var2 = var1.strip(), var2.strip()
            if var1 in variable_names and var2 in variable_names and var1 != var2:
                # Store as sorted tuple to ensure uniqueness
                valid_pairs.add(tuple(sorted([var1, var2])))
        
        return True, list(valid_pairs)
        
    except Exception as e:
        return False, f"Parsing error: {str(e)}"

def construct_dag_from_skeleton(edge_pairs, variable_names, causal_ordering):
    """Construct directed adjacency matrix from skeleton and causal ordering."""
    n = len(variable_names)
    var_to_idx = {var: i for i, var in enumerate(variable_names)}
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # Create ordering position map
    ordering_pos = {var_to_idx[variable_names[i]]: i for i in range(len(causal_ordering))}
    
    for var1, var2 in edge_pairs:
        idx1, idx2 = var_to_idx[var1], var_to_idx[var2]
        
        # Orient edge based on causal ordering
        if ordering_pos[idx1] < ordering_pos[idx2]:
            adj_matrix[idx1][idx2] = 1  # var1 -> var2
        else:
            adj_matrix[idx2][idx1] = 1  # var2 -> var1
    
    return adj_matrix

def create_prompt(variable_names):
    """Create the prompt with randomized variable names using template."""
    # Randomize variable order
    shuffled_variables = variable_names.copy()
    random.shuffle(shuffled_variables)
    
    var_names_str = ", ".join(shuffled_variables)
    
    # Load prompt template
    prompt_template = load_prompt(PROMPT_FORMAT)
    
    # Format template with variables
    prompt = prompt_template.format(var_names_str=var_names_str)
    
    return prompt, shuffled_variables

# parse_edge_pairs function is defined above with other skeleton functions

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
    """Run the complete skeleton-based experiment."""
    print("Starting LLM Skeleton Guess Experiment...")
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    adj_matrix, data, variable_names = np.load(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                     "real-world-data", f"{DATASET_NAME}-real-world-data.npy"), 
        allow_pickle=True
    )
    variable_names = list(variable_names)
    print(f"Variable names: {variable_names}")
    
    # Get causal ordering from ground truth
    causal_ordering = get_causal_ordering(adj_matrix)
    print(f"Causal ordering: {[variable_names[i] for i in causal_ordering]}")
    
    # Create experiment directory
    exp_num = find_next_experiment_number()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(project_root, "guess-llms-results", f"guess_experiment_{exp_num}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    
    # Save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "prompt_format": PROMPT_FORMAT,
        "num_trials": NUM_TRIALS,
        "variable_names": variable_names,
        "causal_ordering": [variable_names[i] for i in causal_ordering],
        "experiment_type": "skeleton_guess",
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
            
            # Create prompt with randomized variables for this trial
            prompt, shuffled_variables = create_prompt(variable_names)
            
            for retry in range(max_retries + 1):
                try:
                    response, wait_time, attempts = call_bedrock_with_retry(
                        prompt, 
                        API_KEY, 
                        model_info["model_id"], 
                        max_tokens=MAX_TOKENS,
                        region=model_info.get("region", "us-east-1")
                    )
                    
                    # Try to parse edge pairs with detailed error reporting
                    parse_success, edges_or_error = parse_edge_pairs(response, variable_names)
                    
                    if parse_success:
                        # Construct DAG from skeleton and causal ordering
                        dag_matrix = construct_dag_from_skeleton(edges_or_error, variable_names, causal_ordering)
                        
                        # Validate that the constructed matrix is a valid DAG
                        if is_valid_dag(dag_matrix.tolist()):
                            trial_result = {
                                "trial_number": trial + 1,
                                "response": response,
                                "shuffled_variables": shuffled_variables,
                                "edge_pairs": edges_or_error,
                                "adjacency_matrix": dag_matrix.tolist(),
                                "success": True,
                                "error": None,
                                "wait_time": wait_time,
                                "attempts": attempts,
                                "parsing_retries": retry
                            }
                            
                            print(f"    ✓ Success after {attempts} attempts - Found {len(edges_or_error)} edges")
                            trial_success = True
                            break
                        else:
                            # DAG validation failed
                            if retry < max_retries:
                                print(f"    ✗ DAG_ERROR: Constructed matrix is not a valid DAG - Retry {retry + 1}/{max_retries}")
                            else:
                                print(f"    ✗ FINAL FAILURE: DAG_ERROR: Constructed matrix is not a valid DAG")
                                trial_result = {
                                    "trial_number": trial + 1,
                                    "response": response,
                                    "shuffled_variables": shuffled_variables,
                                    "edge_pairs": edges_or_error,
                                    "adjacency_matrix": None,
                                    "success": False,
                                    "error": "DAG_ERROR: Constructed matrix is not a valid DAG",
                                    "detailed_error": "Matrix validation failed after skeleton reconstruction",
                                    "wait_time": wait_time,
                                    "attempts": attempts,
                                    "parsing_retries": retry
                                }
                    else:
                        # Detailed error reporting
                        error_msg = str(edges_or_error)
                        if "No EDGES: tag found" in error_msg:
                            failure_type = "PARSING_ERROR: No EDGES tag detected"
                        elif "No valid edge pairs found" in error_msg:
                            failure_type = "PARSING_ERROR: No valid edge pairs extracted"
                        elif "Parsing error" in error_msg:
                            failure_type = "PARSING_ERROR: Edge extraction failed"
                        else:
                            failure_type = f"UNKNOWN_ERROR: {error_msg}"
                        
                        if retry < max_retries:
                            print(f"    ✗ {failure_type} - Retry {retry + 1}/{max_retries}")
                        else:
                            print(f"    ✗ FINAL FAILURE: {failure_type}")
                            trial_result = {
                                "trial_number": trial + 1,
                                "response": response,
                                "shuffled_variables": shuffled_variables,
                                "edge_pairs": None,
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
                            "shuffled_variables": None,
                            "edge_pairs": None,
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
    
    print(f"\nSkeleton Experiment {exp_num} completed! Results saved to: {exp_dir}")

if __name__ == "__main__":
    run_experiment()