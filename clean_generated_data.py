import json
import os
import subprocess
import tempfile

INPUT_FILE = "generated_manifests_output/problematic_dataset.json"
OUTPUT_FILE = "generated_manifests_output/cleaned_problematic_dataset.json"
KUBE_LINTER_PATH = "kube-linter" # Assumes kube-linter is in PATH

def check_with_linter(manifest_content_str, original_file_path_for_logging):
    """Runs kube-linter on the given manifest string and returns problem status and output."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml', encoding='utf-8') as tmp_file:
            tmp_file.write(manifest_content_str)
            tmp_file_path = tmp_file.name
        
        # Run kube-linter
        # The check=False is important as kube-linter exits non-zero for legitimate findings
        result = subprocess.run(
            [KUBE_LINTER_PATH, "lint", tmp_file_path],
            capture_output=True, text=True, check=False, encoding='utf-8'
        )
        os.unlink(tmp_file_path) # Clean up temp file

        # kube-linter exits with non-zero if issues are found.
        is_problematic = result.returncode != 0
        linter_output = result.stdout + result.stderr # Combine stdout and stderr for full output
        return is_problematic, linter_output

    except FileNotFoundError:
        print(f"Error: kube-linter command not found at '{KUBE_LINTER_PATH}'. Ensure it's installed and in PATH.")
        # If linter can't run, we can't confirm, so treat as not confirmed problematic by this script.
        return False, "Linter not found, could not verify."
    except Exception as e:
        print(f"Error running kube-linter for a variant (originally from {original_file_path_for_logging}): {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return False, f"Linter check failed: {e}"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input JSON file {INPUT_FILE}: {e}")
        return

    print(f"Loaded {len(data)} manifests from {INPUT_FILE}")
    cleaned_manifests = []
    processed_count = 0

    for entry in data:
        processed_count += 1
        print(f"Processing entry {processed_count}/{len(data)} (Original: {entry.get('original_file', 'N/A')}, Mod: {entry.get('modifications', 'N/A')})")
        yaml_content = entry.get("problematic_yaml")
        original_file = entry.get("original_file", "Unknown original file")

        if not yaml_content:
            print(f"  Skipping entry due to missing 'problematic_yaml'.")
            continue

        is_problematic, linter_output = check_with_linter(yaml_content, original_file)

        if is_problematic:
            entry["linter_output"] = linter_output
            cleaned_manifests.append(entry)
            print(f"  KEPT: Linter found issues for variant from {original_file}.")
        else:
            print(f"  DISCARDED: Linter found no issues (or failed) for variant from {original_file}. Output:\n{linter_output[:500]}...") # Print snippet of linter output for discarded items

    print(f"Finished processing. Kept {len(cleaned_manifests)} out of {len(data)} manifests.")

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(cleaned_manifests, f, indent=2)
        print(f"Successfully saved cleaned and verified problematic manifests to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing output JSON file {OUTPUT_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing JSON output: {e}")

if __name__ == "__main__":
    main() 