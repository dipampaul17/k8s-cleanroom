import subprocess
import json
import yaml
import tempfile
import os
import openai
import sys

# Initialize OpenAI client. Assumes OPENAI_API_KEY is set in the environment.
try:
    client = openai.OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.", file=sys.stderr)
    client = None

MAX_YAML_SNIPPET_LINES = 50
# MAX_ISSUES_JSON_LEN no longer needed for LLM prompt,kube-linter output is handled directly
LLM_MAX_OUTPUT_TOKENS = 150 # Max tokens for the assessment_reason string
KUBE_LINTER_CONFIG_PATH = ".kube-linter.yaml"
MAX_EXAMPLE_CHECKS_TO_LLM = 3 # Max number of example check names of each category for the LLM reason prompt

def truncate_string(text, max_len):
    return text if len(text) <= max_len else text[:max_len-3] + "..."

def get_yaml_snippet(yaml_str, max_lines):
    lines = yaml_str.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n... (truncated)"
    return yaml_str

def evaluate_manifest(original_yaml, fixed_yaml):
    """
    Evaluate if the fixed manifest resolves security issues using kube-linter.
    LLM assists in generating a human-readable assessment reason.
    """
    try:
        # Use yaml.safe_load_all() to handle potential multi-document YAML strings
        documents = list(yaml.safe_load_all(fixed_yaml))
        if not documents:
            # No documents found in the YAML string
            # Raise YAMLError to be caught by the existing handler
            raise yaml.YAMLError("No YAML documents found in the provided fixed manifest.")
        
        # We'll process the first document for grading purposes
        fixed_yaml_data = documents[0] 

        if fixed_yaml_data is None: 
            # The first document parsed to None (e.g. an empty document "---")
            # Raise YAMLError to be caught by the existing handler
            raise yaml.YAMLError("The first YAML document is empty or null after parsing.")

    except yaml.YAMLError as e:
        return {
            "reward": 0.0, 
            "reason": f"Invalid fixed YAML: {e}", 
            "original_criticals": 0, 
            "remaining_criticals": 0, 
            "newly_introduced_criticals": 0,
            "is_valid_yaml": False
        }

    # IMPORTANT: The rest of the function expects `fixed_yaml` (the string) 
    # to be written to a temp file for kube-linter.
    # If we only grade the first document, we should ideally only pass the first document
    # (re-serialized to string) to kube-linter. Otherwise, kube-linter might still see all documents.
    # For now, let's write the original `fixed_yaml` which might contain multiple documents
    # to the temp file. Kube-linter itself can process multi-document files.
    # The key part was `yaml.safe_load` for initial validation and potentially for `fixed_yaml_data` usage
    # if it were used beyond simple validation. The current code doesn't seem to use `fixed_yaml_data` variable further.

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as orig_file_obj, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as fixed_file_obj:
        orig_file_name = orig_file_obj.name
        fixed_file_name = fixed_file_obj.name
        orig_file_obj.write(original_yaml)
        fixed_file_obj.write(fixed_yaml)
    
    actual_original_critical_count = 0
    actual_fixed_critical_count = 0
    orig_criticals_list = []
    fixed_criticals_list = []

    try:
        orig_issues = run_kube_linter(orig_file_name)
        orig_criticals_list = list(orig_issues) if orig_issues else []
        actual_original_critical_count = len(orig_criticals_list)
        
        fixed_issues = run_kube_linter(fixed_file_name)
        fixed_criticals_list = list(fixed_issues) if fixed_issues else []
        actual_fixed_critical_count = len(fixed_criticals_list)
    finally:
        os.remove(orig_file_name)
        os.remove(fixed_file_name)

    # Python-side calculation of new, resolved, and remaining check names
    orig_check_names_set = set([issue.get("Check") for issue in orig_criticals_list if issue.get("Check")])
    fixed_check_names_set = set([issue.get("Check") for issue in fixed_criticals_list if issue.get("Check")])

    newly_introduced_check_names = sorted(list(fixed_check_names_set - orig_check_names_set))
    resolved_check_names = sorted(list(orig_check_names_set - fixed_check_names_set))
    persisting_check_names = sorted(list(orig_check_names_set.intersection(fixed_check_names_set)))
    
    # For MVP, newly_introduced_criticals count is based on new check *types*
    newly_introduced_criticals_count_py = len(newly_introduced_check_names)

    llm_assessment_reason = "Assessment reason generation skipped or failed."
    if client: # Only attempt LLM call if client is initialized
        prompt_template_llm = """
You are an expert Kubernetes security auditor. 
Your task is to provide a concise, human-readable assessment string (1-2 sentences) based on the summary of changes to critical security issues.

**Input Data:**
- `actual_original_critical_count`: {actual_original_critical_count} (total critical issues in original)
- `actual_fixed_critical_count`: {actual_fixed_critical_count} (total critical issues in fixed manifest)
- `newly_introduced_criticals_count_py`: {newly_introduced_criticals_count_py} (count of new *types* of critical issues introduced by the fix)
- `example_resolved_check_names`: {example_resolved_check_names} (sample of up to {MAX_EXAMPLE_CHECKS_TO_LLM} resolved issue types)
- `example_new_check_names`: {example_new_check_names} (sample of up to {MAX_EXAMPLE_CHECKS_TO_LLM} new issue types)
- `example_persisting_check_names`: {example_persisting_check_names} (sample of up to {MAX_EXAMPLE_CHECKS_TO_LLM} persisting issue types)

**Your Task:**
Generate a single, brief assessment string. Focus on the overall impact. 
Examples:
- "Fix successfully addressed {len_resolved} types of issues, including '{first_resolved_example}', with no new issue types introduced. {actual_fixed_critical_count} issues remain."
- "While {len_resolved} issue types like '{first_resolved_example}' were resolved, {newly_introduced_criticals_count_py} new types of issues such as '{first_new_example}' were introduced. Overall, critical issues changed from {actual_original_critical_count} to {actual_fixed_critical_count}."
- "No critical issues in original. Fix introduced {newly_introduced_criticals_count_py} new issue types (e.g., '{first_new_example}')."
- "Manifest remains clean with no critical issues detected."
- "Original manifest had {actual_original_critical_count} issues. The fix resulted in {actual_fixed_critical_count} issues, including {newly_introduced_criticals_count_py} new types like '{first_new_example}' but resolved {len_resolved} types like '{first_resolved_example}'."

Do NOT output JSON. Just the assessment string.
"""
        
        formatted_prompt_llm = prompt_template_llm.format(
            actual_original_critical_count=actual_original_critical_count,
            actual_fixed_critical_count=actual_fixed_critical_count,
            newly_introduced_criticals_count_py=newly_introduced_criticals_count_py,
            example_resolved_check_names=json.dumps(resolved_check_names[:MAX_EXAMPLE_CHECKS_TO_LLM]),
            example_new_check_names=json.dumps(newly_introduced_check_names[:MAX_EXAMPLE_CHECKS_TO_LLM]),
            example_persisting_check_names=json.dumps(persisting_check_names[:MAX_EXAMPLE_CHECKS_TO_LLM]),
            MAX_EXAMPLE_CHECKS_TO_LLM=MAX_EXAMPLE_CHECKS_TO_LLM,
            len_resolved=len(resolved_check_names),
            first_resolved_example=resolved_check_names[0] if resolved_check_names else "N/A",
            first_new_example=newly_introduced_check_names[0] if newly_introduced_check_names else "N/A"
        )

        try:
            response = client.responses.create(
                model="o4-mini",
                reasoning={"effort": "low"}, 
                input=[{"role": "user", "content": formatted_prompt_llm}],
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS
            )
            if response.status == "incomplete" and response.incomplete_details and response.incomplete_details.reason == "max_output_tokens":
                llm_assessment_reason = "LLM summary generation was cut short due to token limits."
            elif response.output_text and response.output_text.strip():
                llm_assessment_reason = response.output_text.strip().replace('\n', ' ') # Ensure single line
            else:
                 llm_assessment_reason = "LLM provided no assessment text; relying on default summary."

        except openai.APIError as e:
            print(f"OpenAI API error: {e}", file=sys.stderr)
            llm_assessment_reason = f"OpenAI API error during assessment reason generation: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during LLM reason generation: {e}", file=sys.stderr)
            llm_assessment_reason = f"Unexpected error during LLM reason generation: {e}"
    else:
        llm_assessment_reason = "OpenAI client not initialized. Cannot generate LLM assessment reason."

    # Construct final reason string using LLM output or a Python-generated one if LLM failed
    final_reason = llm_assessment_reason
    if "failed" in final_reason.lower() or "error" in final_reason.lower() or "skipped" in final_reason.lower() or "cut short" in final_reason.lower() or "no assessment text" in final_reason.lower() or "not initialized" in final_reason.lower() :
        # Fallback to a Python-generated reason if LLM part was problematic
        reason_parts = []
        if actual_original_critical_count == 0 and newly_introduced_criticals_count_py == 0 and actual_fixed_critical_count == 0:
            reason_parts.append("Manifest was and remains clean.")
        elif actual_original_critical_count > 0 and actual_fixed_critical_count == 0 and newly_introduced_criticals_count_py == 0:
            reason_parts.append(f"All {actual_original_critical_count} original critical issue types fixed successfully.")
        else:
            reason_parts.append(f"Original issues: {actual_original_critical_count}.")
            reason_parts.append(f"Fixed issues: {actual_fixed_critical_count}.")
            if newly_introduced_criticals_count_py > 0:
                reason_parts.append(f"{newly_introduced_criticals_count_py} new issue types (e.g., {newly_introduced_check_names[0] if newly_introduced_check_names else 'N/A'}) were introduced.")
            if len(resolved_check_names) > 0:
                 reason_parts.append(f"{len(resolved_check_names)} issue types (e.g., {resolved_check_names[0] if resolved_check_names else 'N/A'}) were resolved.")
        final_reason = "Summary: " + " ".join(reason_parts)
        if final_reason == "Summary: ": # Catch if no parts were added
             final_reason = f"Evaluation complete. Counts: Original={actual_original_critical_count}, Fixed={actual_fixed_critical_count}, New Types={newly_introduced_criticals_count_py}."

    # Python-side reward calculation (uses the Python-calculated newly_introduced_criticals_count_py)
    reward = 0.0
    if actual_original_critical_count == 0:
        if newly_introduced_criticals_count_py > 0:
            reward = 0.0 # Penalize introducing issues to a clean manifest
        else:
            reward = 1.0 # Manifest was clean and remains clean
    else: # actual_original_critical_count > 0
        if actual_fixed_critical_count == 0 and newly_introduced_criticals_count_py == 0:
            reward = 1.0 # All issues fixed, no new ones
        else:
            # Consider new issue types as negating some of the fixes
            improvement_score_numerator = actual_original_critical_count - actual_fixed_critical_count - newly_introduced_criticals_count_py
            improvement_score = improvement_score_numerator / actual_original_critical_count
            reward = max(0.0, min(0.8, improvement_score)) # Cap partial fixes at 0.8

    return {
        "reward": float(reward),
        "original_criticals": actual_original_critical_count,
        "remaining_criticals": actual_fixed_critical_count,
        "newly_introduced_criticals": newly_introduced_criticals_count_py, # Using Python-calculated value
        "is_valid_yaml": True,
        "reason": final_reason.strip()
    }

def run_kube_linter(filename):
    """Run kube-linter with config and parse output. Returns list of reports or empty list."""
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
            timeout=10 
        )
        if result.returncode != 0 and not result.stdout.strip():
            print(f"kube-linter failed for {filename} with exit code {result.returncode}. Stderr: {result.stderr.strip() if result.stderr else 'N/A'}", file=sys.stderr)
            return []
        if not result.stdout.strip():
            return [] 
        try:
            parsed_output = json.loads(result.stdout)
            return parsed_output.get('Reports', [])
        except json.JSONDecodeError as je:
            print(f"Error decoding kube-linter JSON output for {filename}: {je}. Output snippet: {result.stdout[:500]}", file=sys.stderr)
            return []
    except FileNotFoundError:
        print(f"CRITICAL ERROR: kube-linter command not found or config {KUBE_LINTER_CONFIG_PATH} not found. Ensure kube-linter is installed and in PATH, and config is copied in Docker.", file=sys.stderr)
        raise
    except subprocess.TimeoutExpired:
        print(f"kube-linter timed out processing {filename}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Unexpected error running kube-linter for {filename}: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if not client:
        print("Exiting: OpenAI client failed to initialize. Ensure OPENAI_API_KEY is set.", file=sys.stderr)
        sys.exit(1)
        
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            with open(filepath, 'r') as f:
                original_yaml_str = f.read()
            print(f"Evaluating manifest: {filepath}")
            
            fixed_yaml_str = original_yaml_str
            try:
                original_data = yaml.safe_load(original_yaml_str)
                fixed_data = yaml.safe_load(original_yaml_str) 

                modified = False
                if isinstance(fixed_data, dict) and fixed_data.get("kind") == "Pod":
                    containers = fixed_data.get("spec", {}).get("containers", [])
                    for container in containers:
                        if isinstance(container.get("securityContext"), dict):
                            if container["securityContext"].get("privileged") == True:
                                container["securityContext"]["privileged"] = False
                                print("Test fix: Set privileged to false")
                                modified = True
                            if container["securityContext"].get("runAsUser") == 0:
                                container["securityContext"]["runAsUser"] = 1001
                                print("Test fix: Set runAsUser to 1001")
                                modified = True
                        if container.get("image", "").endswith(":latest") or ":" not in container.get("image", ""):
                             if container.get("image", "").startswith("busybox"):
                                 container["image"] = "busybox:1.36"
                                 print("Test fix: Changed busybox image to busybox:1.36")
                                 modified = True               
                if modified:
                    fixed_yaml_str = yaml.dump(fixed_data)
                else:
                    print("Could not apply a simple programmatic fix for testing; evaluating original against itself or a minor variant.")
                    if "replicas: 1" in original_yaml_str:
                         fixed_yaml_str = original_yaml_str.replace("replicas: 1", "replicas: 2", 1)
            except Exception as e:
                print(f"Error trying to create a fixed test version: {e}. Evaluating original against itself.", file=sys.stderr)

            print("--- Original Snippet ---")
            print(get_yaml_snippet(original_yaml_str, 10))
            print("--- Fixed Snippet (Test) ---")
            print(get_yaml_snippet(fixed_yaml_str, 10))
            print("------------------------")

            result = evaluate_manifest(original_yaml_str, fixed_yaml_str)
            print(json.dumps(result, indent=2))

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred in the main execution block: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    else:
        print("Usage: python grader.py <path_to_original_yaml>")
        print("\nRunning with dummy data as an example (no file provided):")
        dummy_original_yaml = """ 
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment-orig
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test-orig
  template:
    metadata:
      labels:
        app: test-orig
    spec:
      containers:
      - name: main
        image: nginx:latest # Kube-linter: latest-tag
        securityContext:
          runAsUser: 0 # Kube-linter: run-as-non-root
          allowPrivilegeEscalation: true # Kube-linter: privilege-escalation-container
""" 
        dummy_fixed_yaml_perfect = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment-fixed-perfect
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test-fixed-perfect
  template:
    metadata:
      labels:
        app: test-fixed-perfect
    spec:
      containers:
      - name: main
        image: nginx:1.25.4
        securityContext:
          runAsUser: 1001
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
"""
        dummy_fixed_yaml_partial_new_issue = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment-fixed-partial
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test-fixed-partial
  template:
    metadata:
      labels:
        app: test-fixed-partial
    spec:
      containers:
      - name: main
        image: nginx:1.25.4 # Original 'latest-tag' fixed
        securityContext:
          runAsUser: 0 # Original 'run-as-non-root' persists
          # allowPrivilegeEscalation: true # Original 'privilege-escalation-container' fixed
          privileged: true # New issue for 'privileged-container' and also 'privilege-escalation-container' if not already from allowPrivilegeEscalation
"""
        dummy_orig_clean = """
apiVersion: v1
kind: Pod
metadata:
  name: clean-pod
spec:
  containers:
  - name: main
    image: nginx:1.25.4
    securityContext:
      runAsUser: 1001
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
"""
        dummy_fixed_clean_adds_issue = """
apiVersion: v1
kind: Pod
metadata:
  name: clean-pod-adds-issue
spec:
  containers:
  - name: main
    image: nginx:1.25.4
    securityContext:
      runAsUser: 0 # New issue: run-as-non-root
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
"""

        print("\n--- Evaluating: dummy_original_yaml vs dummy_fixed_yaml_perfect ---")
        result_good = evaluate_manifest(dummy_original_yaml, dummy_fixed_yaml_perfect)
        print(json.dumps(result_good, indent=2))
        
        print("\n--- Evaluating: dummy_original_yaml vs dummy_fixed_yaml_partial_new_issue ---")
        result_partial_new = evaluate_manifest(dummy_original_yaml, dummy_fixed_yaml_partial_new_issue)
        print(json.dumps(result_partial_new, indent=2))

        print("\n--- Evaluating: dummy_orig_clean vs dummy_fixed_clean_adds_issue ---")
        result_clean_adds_issue = evaluate_manifest(dummy_orig_clean, dummy_fixed_clean_adds_issue)
        print(json.dumps(result_clean_adds_issue, indent=2))

        print("\n--- Evaluating: dummy_orig_clean vs dummy_orig_clean (no changes) ---")
        result_clean_no_change = evaluate_manifest(dummy_orig_clean, dummy_orig_clean)
        print(json.dumps(result_clean_no_change, indent=2)) 