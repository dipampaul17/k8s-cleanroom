import os
import yaml
import json
import random
import subprocess
import copy
import tempfile # Ensure tempfile is imported at the top

# Configuration
# Using paths relative to the workspace root /Users/dipampaul_/Downloads/k8s
BASE_SCAN_DIRECTORIES = [
    "k8s-cleanroom/data/examples",
    "k8s-cleanroom/data/kubebuilder-declarative-pattern/examples"
]
REPOS_TO_CLONE = {
    "examples": "https://github.com/kubernetes/examples.git",
    "argo-cd": "https://github.com/argoproj/argo-cd.git",
    "cert-manager": "https://github.com/cert-manager/cert-manager.git"
}
CLONED_REPOS_DIR = "cloned_repos"

# Specific subdirectories within cloned repos to scan
REPO_SPECIFIC_PATHS = {
    "examples": [""], # Scan the whole repo
    "argo-cd": ["manifests/core-install", "manifests/ha", "manifests/install.yaml"], # Specific manifest paths/files
    "cert-manager": ["deploy/charts", "deploy/yaml", "deploy/crds"] # Scan these subdirectories
}

OUTPUT_FILE = "generated_manifests_output/problematic_dataset.json"
TARGET_EXAMPLES = 500
KUBE_LINTER_PATH = "kube-linter" # Assumes kube-linter is in PATH

def clone_repositories():
    if not os.path.exists(CLONED_REPOS_DIR):
        os.makedirs(CLONED_REPOS_DIR)
        print(f"Created directory: {CLONED_REPOS_DIR}")

    cloned_paths_to_scan = []

    for repo_name, repo_url in REPOS_TO_CLONE.items():
        repo_path = os.path.join(CLONED_REPOS_DIR, repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo_url} into {repo_path}...")
            try:
                subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path], check=True, capture_output=True, text=True)
                print(f"Successfully cloned {repo_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error cloning {repo_name}: {e.stderr}")
                continue # Skip this repo if cloning fails
        else:
            print(f"Repository {repo_name} already exists at {repo_path}. Skipping clone.")
            # Optionally, add logic here to git pull for updates if desired
            # print(f"Attempting to update {repo_name}...")
            # try:
            #     subprocess.run(["git", "-C", repo_path, "pull"], check=True, capture_output=True, text=True)
            #     print(f"Successfully updated {repo_name}.")
            # except subprocess.CalledProcessError as e:
            #     print(f"Error updating {repo_name}: {e.stderr}")

        # Add specified paths for this repo to scan list
        if repo_name in REPO_SPECIFIC_PATHS:
            for specific_path_or_file in REPO_SPECIFIC_PATHS[repo_name]:
                full_path = os.path.join(repo_path, specific_path_or_file)
                if os.path.exists(full_path):
                    cloned_paths_to_scan.append(full_path)
                else:
                    print(f"Warning: Specified path/file for {repo_name} not found: {full_path}")
        else: # If no specific paths, add the whole repo dir
             cloned_paths_to_scan.append(repo_path)


    return cloned_paths_to_scan

def find_yaml_files(directories_or_files):
    yaml_files = []
    for path_item in directories_or_files:
        if os.path.isdir(path_item):
            abs_directory = os.path.join(os.getcwd(), path_item)
            if not os.path.isdir(abs_directory):
                print(f"Warning: Directory not found or not a directory: {abs_directory}")
                continue
            for root, _, files in os.walk(abs_directory):
                for file in files:
                    if file.endswith((".yaml", ".yml")):
                        yaml_files.append(os.path.join(root, file))
        elif os.path.isfile(path_item) and path_item.endswith((".yaml", ".yml")):
            yaml_files.append(os.path.join(os.getcwd(), path_item))
        elif os.path.isfile(path_item): # A specific file was given that is not yaml, skip
             print(f"Warning: Specified file is not a YAML file, skipping: {path_item}")
        else:
            print(f"Warning: Path item not found or not a file/directory: {path_item}")


    return list(set(yaml_files)) # Remove duplicates

def load_manifests(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
            # Use load_all to handle multi-document YAML files
            manifest_docs = list(yaml.safe_load_all(f))
            # Filter out None results from empty documents or comments-only sections
            return [doc for doc in manifest_docs if doc is not None]
    except yaml.YAMLError as ye:
        print(f"YAML parsing error in file {file_path}: {ye}")
        return []
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return []

def add_privileged(manifest):
    """Adds privileged: true to the first container's securityContext."""
    modified_manifest = copy.deepcopy(manifest)
    if not isinstance(modified_manifest, dict): return None, []
    
    applied_change_desc = "add_privileged"
    changed = False
    
    if modified_manifest.get("kind") in ["Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet", "ReplicationController"]:
        spec = modified_manifest.get("spec")
        if isinstance(spec, dict):
            template = spec.get("template") # For Deployments, StatefulSets etc.
            if isinstance(template, dict) and isinstance(template.get("spec"), dict): # Pod template
                pod_spec = template.get("spec")
            elif modified_manifest.get("kind") == "Pod": # Direct Pod spec
                 pod_spec = spec
            else: # Job/CronJob might have Pod template
                job_template = spec.get("jobTemplate")
                if isinstance(job_template, dict) and isinstance(job_template.get("spec"), dict) and isinstance(job_template.get("spec").get("template"), dict) and isinstance(job_template.get("spec").get("template").get("spec"), dict):
                     pod_spec = job_template.get("spec").get("template").get("spec")
                elif isinstance(spec.get("template"), dict) and isinstance(spec.get("template").get("spec"), dict): 
                    pod_spec = spec.get("template").get("spec")
                else:
                    pod_spec = None

            if isinstance(pod_spec, dict):
                containers = pod_spec.get("containers", [])
                init_containers = pod_spec.get("initContainers", [])
                all_containers = containers + init_containers

                for container in all_containers:
                    if isinstance(container, dict):
                        if "securityContext" not in container:
                            container["securityContext"] = {}
                        if isinstance(container["securityContext"], dict):
                            if container["securityContext"].get("privileged") is not True:
                                container["securityContext"]["privileged"] = True
                                changed = True
                                # Apply to first container found in either list and break, or all?
                                # For more vulns, let's try to make it true on first applicable one.
                                break 
                if changed: # Ensure only one change is reported if multiple containers exist.
                     pass


    return modified_manifest if changed else None, [applied_change_desc] if changed else []

def remove_resource_limits(manifest):
    """Removes resource limits from all containers."""
    modified_manifest = copy.deepcopy(manifest)
    if not isinstance(modified_manifest, dict): return None, []

    applied_change_desc = "remove_resource_limits"
    changed = False

    def _process_pod_spec(pod_spec_dict):
        nonlocal changed
        if isinstance(pod_spec_dict, dict):
            all_containers = pod_spec_dict.get("containers", []) + pod_spec_dict.get("initContainers", [])
            for container in all_containers:
                if isinstance(container, dict) and isinstance(container.get("resources"), dict):
                    if "limits" in container["resources"]:
                        del container["resources"]["limits"]
                        changed = True
                    if not container["resources"]: 
                        del container["resources"]
                        changed = True # Also flag if resources dict becomes empty


    if modified_manifest.get("kind") in ["Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet", "ReplicationController"]:
        spec = modified_manifest.get("spec")
        if isinstance(spec, dict):
            template = spec.get("template")
            if isinstance(template, dict) and isinstance(template.get("spec"), dict):
                _process_pod_spec(template.get("spec"))
            elif modified_manifest.get("kind") == "Pod":
                _process_pod_spec(spec)
            else:
                job_template = spec.get("jobTemplate")
                if isinstance(job_template, dict) and isinstance(job_template.get("spec"), dict) and isinstance(job_template.get("spec").get("template"), dict) and isinstance(job_template.get("spec").get("template").get("spec"), dict):
                     _process_pod_spec(job_template.get("spec").get("template").get("spec"))
                elif isinstance(spec.get("template"), dict) and isinstance(spec.get("template").get("spec"), dict):
                    _process_pod_spec(spec.get("template").get("spec"))

    return modified_manifest if changed else None, [applied_change_desc] if changed else []


def add_hostpath_volume(manifest):
    """Adds a hostPath volume and mounts it to the first container."""
    modified_manifest = copy.deepcopy(manifest)
    if not isinstance(modified_manifest, dict): return None, []

    applied_change_desc = "add_hostpath_volume"
    changed = False
    host_path_volume_name = "dangerous-hostpath"

    if modified_manifest.get("kind") in ["Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet", "ReplicationController"]:
        spec = modified_manifest.get("spec")
        if isinstance(spec, dict):
            target_pod_spec = None
            template = spec.get("template")
            if isinstance(template, dict) and isinstance(template.get("spec"), dict):
                target_pod_spec = template.get("spec")
            elif modified_manifest.get("kind") == "Pod":
                target_pod_spec = spec
            else: 
                job_template = spec.get("jobTemplate")
                if isinstance(job_template, dict) and isinstance(job_template.get("spec"), dict) and isinstance(job_template.get("spec").get("template"), dict) and isinstance(job_template.get("spec").get("template").get("spec"), dict):
                     target_pod_spec = job_template.get("spec").get("template").get("spec")
                elif isinstance(spec.get("template"), dict) and isinstance(spec.get("template").get("spec"), dict):
                    target_pod_spec = spec.get("template").get("spec")

            if isinstance(target_pod_spec, dict):
                if "volumes" not in target_pod_spec:
                    target_pod_spec["volumes"] = []
                
                volume_exists = any(v.get("name") == host_path_volume_name for v in target_pod_spec["volumes"] if isinstance(v,dict))

                if not volume_exists:
                    target_pod_spec["volumes"].append({
                        "name": host_path_volume_name,
                        "hostPath": {"path": "/tmp/danger-" + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6)), "type": "DirectoryOrCreate"} # Unique path
                    })
                    changed_volume_add = True

                    # Mount volume in first container (either regular or init)
                    all_containers = target_pod_spec.get("containers", []) + target_pod_spec.get("initContainers", [])
                    if all_containers and isinstance(all_containers[0], dict):
                        first_container = all_containers[0]
                        if "volumeMounts" not in first_container:
                            first_container["volumeMounts"] = []
                        
                        mount_exists = any(vm.get("name") == host_path_volume_name for vm in first_container["volumeMounts"] if isinstance(vm,dict))
                        if not mount_exists:
                            first_container["volumeMounts"].append({
                                "name": host_path_volume_name,
                                "mountPath": "/mnt/host-" + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6)) # Unique mount
                            })
                            changed = True # Final change flag
                    elif changed_volume_add : # Volume added but no container to mount, still a change
                        changed = True

    return modified_manifest if changed else None, [applied_change_desc] if changed else []

def remove_security_context(manifest):
    """Removes securityContext from pod and container levels."""
    modified_manifest = copy.deepcopy(manifest)
    if not isinstance(modified_manifest, dict): return None, []

    applied_change_desc = "remove_security_context"
    changed = False

    def _process_pod_spec(pod_spec_dict):
        nonlocal changed
        if isinstance(pod_spec_dict, dict):
            if "securityContext" in pod_spec_dict and pod_spec_dict["securityContext"]: # Check if not empty
                del pod_spec_dict["securityContext"]
                changed = True
            
            all_containers = pod_spec_dict.get("containers", []) + pod_spec_dict.get("initContainers", [])
            for container in all_containers:
                if isinstance(container, dict) and "securityContext" in container and container["securityContext"]: # Check if not empty
                    del container["securityContext"]
                    changed = True
            
    if modified_manifest.get("kind") in ["Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet", "ReplicationController"]:
        spec = modified_manifest.get("spec")
        if isinstance(spec, dict):
            template = spec.get("template")
            if isinstance(template, dict) and isinstance(template.get("spec"), dict):
                _process_pod_spec(template.get("spec"))
            elif modified_manifest.get("kind") == "Pod":
                _process_pod_spec(spec)
            else:
                job_template = spec.get("jobTemplate")
                if isinstance(job_template, dict) and isinstance(job_template.get("spec"), dict) and isinstance(job_template.get("spec").get("template"), dict) and isinstance(job_template.get("spec").get("template").get("spec"), dict):
                     _process_pod_spec(job_template.get("spec").get("template").get("spec"))
                elif isinstance(spec.get("template"), dict) and isinstance(spec.get("template").get("spec"), dict):
                    _process_pod_spec(spec.get("template").get("spec"))

    return modified_manifest if changed else None, [applied_change_desc] if changed else []

MODIFICATION_FUNCTIONS = [
    add_privileged,
    remove_resource_limits,
    add_hostpath_volume,
    remove_security_context
]

def check_with_linter(manifest_content_str, original_file_path):
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml', encoding='utf-8') as tmp_file:
            tmp_file.write(manifest_content_str)
            tmp_file_path = tmp_file.name
        
        result = subprocess.run([KUBE_LINTER_PATH, "lint", tmp_file_path], capture_output=True, text=True, check=False, encoding='utf-8')
        os.unlink(tmp_file_path) 

        return result.returncode != 0, result.stdout + result.stderr
    except FileNotFoundError:
        print(f"Error: kube-linter command not found at '{KUBE_LINTER_PATH}'. Please ensure it's installed and in your PATH.")
        return True, "Linter not found, assuming problematic."
    except Exception as e:
        print(f"Error running kube-linter for a variant of {original_file_path}: {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return False, f"Linter check failed: {e}"

def main():
    cloned_scan_paths = clone_repositories()
    all_scan_sources = BASE_SCAN_DIRECTORIES + cloned_scan_paths
    
    print(f"Effective scan sources: {all_scan_sources}")
    yaml_files = find_yaml_files(all_scan_sources)
    print(f"Found {len(yaml_files)} unique YAML files after cloning and combining sources.")

    problematic_manifests = []
    processed_files = 0
    # tempfile is already imported at the top

    for yaml_file_path in yaml_files:
        if len(problematic_manifests) >= TARGET_EXAMPLES:
            print(f"Target of {TARGET_EXAMPLES} problematic manifests reached. Stopping.")
            break
        
        print(f"Processing file ({processed_files+1}/{len(yaml_files)}): {yaml_file_path}")
        original_manifests_in_file = load_manifests(yaml_file_path)
        processed_files += 1

        for original_manifest in original_manifests_in_file:
            if len(problematic_manifests) >= TARGET_EXAMPLES:
                break 
            if not original_manifest or not isinstance(original_manifest, dict) or not original_manifest.get("kind"):
                continue

            # Shuffle functions to try different ones, but try all of them for each manifest
            # random.shuffle(MODIFICATION_FUNCTIONS) # Shuffling can still be good for variety if multiple mods are applied to the same base
            
            for mod_func in MODIFICATION_FUNCTIONS:
                if len(problematic_manifests) >= TARGET_EXAMPLES:
                    break

                # Apply modification to a fresh copy of the original_manifest
                # to ensure modifications are independent for each function.
                current_base_manifest = copy.deepcopy(original_manifest)
                modified_manifest_doc, changes_desc = mod_func(current_base_manifest)
                
                if modified_manifest_doc and changes_desc:
                    try:
                        # Ensure no problematic characters for dump, though default_flow_style=False helps
                        modified_yaml_str = yaml.dump(modified_manifest_doc, sort_keys=False, default_flow_style=False, allow_unicode=True)
                        
                        # Optional: Validate with kube-linter
                        # is_problematic, linter_output = check_with_linter(modified_yaml_str, yaml_file_path)
                        # if is_problematic:
                        problematic_manifests.append({
                            "original_file": yaml_file_path,
                            "modifications": changes_desc, # This will be a list like ['add_privileged']
                            "problematic_yaml": modified_yaml_str,
                            # "linter_output": linter_output 
                        })
                        print(f"  Generated problematic variant from {yaml_file_path} with {changes_desc[0]}. Total: {len(problematic_manifests)}")
                        # DO NOT break here, try other modifications on the same original_manifest
                        # else:
                        #    print(f"  Modification {changes_desc[0]} on {yaml_file_path} did not result in linter issues.")
                    
                    except yaml.YAMLError as ye:
                        print(f"Error serializing modified YAML from {yaml_file_path} (mod: {changes_desc[0]}): {ye}")
                    except Exception as e:
                        print(f"An unexpected error occurred for {yaml_file_path} (mod: {changes_desc[0]}): {e}")
            
    print(f"Generated {len(problematic_manifests)} problematic manifests.")
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: # Specify encoding
            json.dump(problematic_manifests, f, indent=2)
        print(f"Successfully saved problematic manifests to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing output JSON file {OUTPUT_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing JSON output: {e}")

if __name__ == "__main__":
    main() 