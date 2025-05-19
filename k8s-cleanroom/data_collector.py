import os
import subprocess
import glob
import json
import yaml
import random
import tempfile
import sys # For stderr

# Configuration for kube-linter path, assuming .kube-linter.yaml is in the same directory or one level up
# For this script, assume it's run from k8s-cleanroom directory.
KUBE_LINTER_CONFIG_PATH = "./.kube-linter.yaml" 

def run_kube_linter(filename):
    """Run kube-linter with config and parse output. Returns list of reports or empty list."""
    # Check if the config file actually exists at the expected path
    if not os.path.exists(KUBE_LINTER_CONFIG_PATH):
        print(f"CRITICAL ERROR: kube-linter config file not found at {os.path.abspath(KUBE_LINTER_CONFIG_PATH)}. Make sure it exists.", file=sys.stderr)
        # To prevent cascading failures, perhaps return an empty list or raise an error
        # For data collection, we might want to be forgiving and try without config, or just fail.
        # For now, let's try to run without the --config flag if not found, as kube-linter might have defaults.
        # This is a deviation from grader.py but might be more robust for data collection from various states.
        # However, the user's project setup implies the config IS central to how kube-linter is used.
        # Sticking to requiring it for consistency with grader.py's intent:
        raise FileNotFoundError(f"kube-linter config not found at {KUBE_LINTER_CONFIG_PATH}")

    cmd = [
        "kube-linter", "lint",
        "--format", "json",
        "--config", KUBE_LINTER_CONFIG_PATH, 
        filename
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False, 
            timeout=20 # Increased timeout for potentially larger/more complex real-world manifests
        )
        if result.returncode != 0 and not result.stdout.strip():
            # print(f"kube-linter failed for {filename} with exit code {result.returncode}. Stderr: {result.stderr.strip() if result.stderr else 'N/A'}", file=sys.stderr)
            return [] # Suppress stderr for cleaner data collection output
        if not result.stdout.strip():
            return [] 
        try:
            parsed_output = json.loads(result.stdout)
            return parsed_output.get('Reports', [])
        except json.JSONDecodeError as je:
            # print(f"Error decoding kube-linter JSON output for {filename}: {je}. Output snippet: {result.stdout[:200]}", file=sys.stderr)
            return []
    except FileNotFoundError:
        print(f"CRITICAL ERROR: kube-linter command not found. Ensure it is installed and in PATH.", file=sys.stderr)
        # This is a fatal error for the script if kube-linter itself is missing.
        raise 
    except subprocess.TimeoutExpired:
        # print(f"kube-linter timed out processing {filename}", file=sys.stderr)
        return []
    except Exception as e:
        # print(f"Unexpected error running kube-linter for {filename}: {e}", file=sys.stderr)
        return []

def clone_repos():
    """Clone popular K8s repos with manifests"""
    repos = [
        "https://github.com/microservices-demo/microservices-demo.git",
        "https://github.com/kubernetes/examples.git",
        "https://github.com/kubernetes-sigs/kubebuilder-declarative-pattern.git",
        "https://github.com/GoogleCloudPlatform/microservices-demo.git" # Added another similar one for variety
    ]
    
    os.makedirs("data", exist_ok=True)
    for repo_url in repos: # Changed variable name for clarity
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = f"data/{repo_name}"
        if not os.path.exists(repo_path):
            print(f"Cloning {repo_url} into {repo_path}...")
            try:
                subprocess.run(["git", "clone", "--depth=1", repo_url, repo_path], check=True, capture_output=True)
                print(f"Successfully cloned {repo_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone {repo_url}. Error: {e.stderr.decode()}", file=sys.stderr)
        else:
            print(f"Repository {repo_name} already exists in {repo_path}. Skipping clone.")

def find_manifests():
    """Find all YAML files that look like K8s manifests and have issues"""
    manifests = []
    print("Searching for manifests in data/ ...")
    yaml_files = glob.glob("data/**/*.yaml", recursive=True) + glob.glob("data/**/*.yml", recursive=True)
    print(f"Found {len(yaml_files)} total YAML/YML files.")
    
    processed_files = 0
    for yaml_file in yaml_files:
        processed_files +=1
        if processed_files % 50 == 0:
            print(f"Processed {processed_files}/{len(yaml_files)} files for manifest identification...")
        try:
            with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f: # Added encoding and error handling
                content = f.read()
            
            # Basic check if it's a K8s manifest (or part of a multi-document YAML)
            # We check if *any* document in a multi-document YAML is a K8s manifest
            is_k8s_manifest_doc = False
            try:
                all_data = list(yaml.safe_load_all(content)) # Use load_all for multi-document files
            except (yaml.YAMLError, AttributeError, TypeError): # Catch various parsing errors
                # print(f"Skipping {yaml_file} due to YAML parsing error: {e}", file=sys.stderr)
                continue


            if not all_data: # Empty YAML file
                continue

            # Reconstruct content of only the first valid K8s manifest document found for simplicity for now.
            # A more advanced approach would handle each document separately.
            first_k8s_doc_content = None
            for single_data in all_data:
                if isinstance(single_data, dict) and 'apiVersion' in single_data and 'kind' in single_data:
                    is_k8s_manifest_doc = True
                    try:
                        first_k8s_doc_content = yaml.dump(single_data) # Get content of just this doc
                    except yaml.YAMLError:
                        first_k8s_doc_content = None # Could not dump this specific doc
                    break 
            
            if is_k8s_manifest_doc and first_k8s_doc_content:
                # Write the (potentially single doc) content to a temp file to lint
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_lint_file:
                    temp_lint_file.write(first_k8s_doc_content)
                    temp_lint_file_name = temp_lint_file.name
                
                issues = run_kube_linter(temp_lint_file_name)
                os.remove(temp_lint_file_name) # Clean up temp file

                # For data collection, any issue reported by our configured linter is "critical enough"
                if issues: 
                    manifests.append({
                        "path": yaml_file,
                        "content": first_k8s_doc_content, # Store the content of the first K8s doc
                        "issues_found_by_linter": issues # Store raw issues
                    })
        except FileNotFoundError: # If a globbed file was somehow removed
            # print(f"Skipping {yaml_file}, file not found during processing.", file=sys.stderr)
            continue
        except Exception as e:
            # print(f"Skipping {yaml_file} due to unexpected error: {e}", file=sys.stderr)
            continue 
    
    print(f"Found {len(manifests)} initial manifests with issues after filtering.")
    return manifests

# --- Mutation Functions ---
# These functions should accept a Python dictionary (parsed YAML) and return a modified dictionary.

def _get_pod_spec_and_containers(data):
    """Helper to find PodSpec and containers list from common K8s kinds."""
    if not isinstance(data, dict):
        return None, None

    kind = data.get('kind')
    spec = data.get('spec')
    if not spec:
        return None, None

    pod_spec = None
    if kind == 'Pod':
        pod_spec = spec
    elif kind in ['Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
        if isinstance(spec.get('template'), dict) and isinstance(spec['template'].get('spec'), dict):
            pod_spec = spec['template']['spec']
    
    if pod_spec and isinstance(pod_spec.get('containers'), list):
        return pod_spec, pod_spec.get('containers')
    return None, None


def add_privileged(data):
    """Adds privileged: true to the first container's securityContext."""
    pod_spec, containers = _get_pod_spec_and_containers(data)
    if containers:
        for container in containers: # Apply to all containers for more impact
            if not isinstance(container, dict): continue
            if 'securityContext' not in container or container['securityContext'] is None: # Handles if securityContext is None
                container['securityContext'] = {}
            elif not isinstance(container['securityContext'], dict): # Handles if securityContext is not a dict (e.g. boolean)
                 container['securityContext'] = {} # Overwrite if not a dict
            container['securityContext']['privileged'] = True
            # Ensure allowPrivilegeEscalation is also true or absent for privileged to be fully effective
            # Kube-linter often flags privileged without allowPrivilegeEscalation, so let's be explicit.
            container['securityContext']['allowPrivilegeEscalation'] = True 
    return data

def remove_security_context(data):
    """Removes securityContext from containers and pod level."""
    pod_spec, containers = _get_pod_spec_and_containers(data)
    if containers:
        for container in containers:
            if not isinstance(container, dict): continue
            if 'securityContext' in container:
                del container['securityContext']
    if pod_spec and 'securityContext' in pod_spec:
        del pod_spec['securityContext']
    return data

def remove_resource_limits(data):
    """Removes resources.limits from all containers."""
    pod_spec, containers = _get_pod_spec_and_containers(data)
    if containers:
        for container in containers:
            if not isinstance(container, dict): continue
            if 'resources' in container and isinstance(container['resources'], dict) and 'limits' in container['resources']:
                del container['resources']['limits']
            # Also ensure requests are not present, as unset-cpu-requirements etc. often look for both
            if 'resources' in container and isinstance(container['resources'], dict) and 'requests' in container['resources']:
                del container['resources']['requests']
            # If removing limits/requests makes resources empty, remove it too
            if 'resources' in container and isinstance(container['resources'], dict) and not container['resources']:
                del container['resources']

    return data

def add_host_path(data):
    """Adds a hostPath volume (/tmp) and mounts it to all containers."""
    pod_spec, containers = _get_pod_spec_and_containers(data)
    if pod_spec and containers:
        volume_name = "host-tmp-debug" # Unique enough name
        # Ensure volumes list exists
        if 'volumes' not in pod_spec or pod_spec['volumes'] is None:
            pod_spec['volumes'] = []
        elif not isinstance(pod_spec['volumes'], list): # If it exists but not a list
             pod_spec['volumes'] = []


        # Add new hostPath volume, avoid duplicates if this mutation runs multiple times
        if not any(v.get('name') == volume_name for v in pod_spec['volumes'] if isinstance(v, dict)):
            pod_spec['volumes'].append({
                "name": volume_name,
                "hostPath": {"path": "/tmp", "type": "DirectoryOrCreate"} # type is good practice
            })

        for container in containers:
            if not isinstance(container, dict): continue
            if 'volumeMounts' not in container or container['volumeMounts'] is None:
                container['volumeMounts'] = []
            elif not isinstance(container['volumeMounts'], list): # If it exists but not a list
                container['volumeMounts'] = []
            
            # Add new volumeMount, avoid duplicates
            if not any(vm.get('name') == volume_name for vm in container['volumeMounts'] if isinstance(vm, dict)):
                 container['volumeMounts'].append({
                    "name": volume_name,
                    "mountPath": f"/mnt/host-tmp-debug-{container.get('name', 'default').lower()}" # Unique mount path
                })
    return data

# --- End of Mutation Functions ---

def generate_variants(manifests, count=500):
    """Generate synthetic variants with common security issues."""
    all_variants = []
    if not manifests:
        print("No base manifests with issues found to generate variants from.")
        return all_variants
        
    # Use a subset of manifests as templates to avoid overly repetitive variants if source is small
    # Ensure we don't try to pick from an empty list if fewer than 20 manifests are found
    num_base_templates = min(len(manifests), 20)
    if num_base_templates == 0:
        print("Not enough base manifests to generate variants.")
        return all_variants
    base_manifest_templates = random.sample(manifests, num_base_templates)
    
    print(f"Generating up to {count} variants using {num_base_templates} base templates...")
    generated_count = 0

    for i in range(count):
        if generated_count % 50 == 0 and generated_count > 0:
            print(f"Generated {generated_count}/{count} variants...")

        base_manifest_sample = random.choice(base_manifest_templates)
        original_content = base_manifest_sample["content"]
        
        try:
            # We need to load all documents if the original content was multi-doc
            # However, our mutations are designed for single K8s objects.
            # For simplicity, let's assume base_manifest_sample["content"] is a single K8s doc string
            # as per the logic in find_manifests.
            data = yaml.safe_load(original_content) 
            if not isinstance(data, dict): # Skip if not a single dict object
                continue

            mutations = [
                add_privileged,
                remove_security_context,
                remove_resource_limits,
                add_host_path
            ]
            
            num_mutations_to_apply = random.randint(1, min(2, len(mutations))) # Apply 1 or 2 mutations
            chosen_mutations = random.sample(mutations, num_mutations_to_apply)
            
            temp_data = data # Operate on a copy
            for mutation_func in chosen_mutations:
                temp_data = mutation_func(temp_data) # Apply mutation
            
            variant_yaml_str = yaml.dump(temp_data)
            
            # Verify it has issues by writing to a temp file and linting
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_variant_file:
                temp_variant_file.write(variant_yaml_str)
                temp_variant_file_name = temp_variant_file.name
            
            issues = run_kube_linter(temp_variant_file_name)
            os.remove(temp_variant_file_name)
            
            if issues: # We expect mutations to cause issues, or retain original ones
                all_variants.append({
                    "source_path": base_manifest_sample["path"], # Keep track of original source
                    "mutated_content": variant_yaml_str,
                    "linter_issues": issues # Store raw issues
                })
                generated_count += 1
            # else:
                # print(f"Warning: Variant generated from {base_manifest_sample['path']} had no issues after mutation. Skipping.")

        except yaml.YAMLError as e:
            # print(f"YAML error during mutation or dumping for variant based on {base_manifest_sample['path']}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            # print(f"Unexpected error generating variant from {base_manifest_sample['path']}: {e}", file=sys.stderr)
            # import traceback
            # traceback.print_exc() # For debugging
            continue
    
    print(f"Successfully generated {len(all_variants)} variants with issues.")
    return all_variants


def main():
    clone_repos()
    initial_manifests = find_manifests()
    
    if not initial_manifests:
        print("No initial manifests with issues found. Cannot generate variants.")
        return

    variants = generate_variants(initial_manifests, count=500)
    
    output_file_path = "data/training_data.json"
    print(f"Saving {len(variants)} variants to {output_file_path}...")
    with open(output_file_path, 'w') as f:
        json.dump(variants, f, indent=2) # Added indent for readability
    print(f"Data collection complete. Saved to {output_file_path}")

if __name__ == "__main__":
    # Before running main, ensure .kube-linter.yaml exists in the current directory
    # or adjust KUBE_LINTER_CONFIG_PATH if needed.
    if not os.path.exists(KUBE_LINTER_CONFIG_PATH):
        print(f"ERROR: Kube-linter config file '{KUBE_LINTER_CONFIG_PATH}' not found in the current directory ({os.getcwd()}).")
        print("Please ensure the config file is present or update KUBE_LINTER_CONFIG_PATH in the script.")
        sys.exit(1)
    main() 