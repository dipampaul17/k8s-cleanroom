import json
import os
from dotenv import load_dotenv # Added for .env support
from pathlib import Path # Added for explicit path
dotenv_path = Path('.') / '.env' # Explicitly define path in current directory
load_dotenv(dotenv_path=dotenv_path) # Load variables from .env file
print(f"Attempted to load .env from: {dotenv_path.resolve()}") # Debug print
print(f"OPENAI_API_KEY after load_dotenv: {os.getenv('OPENAI_API_KEY') is not None}") # Debug print
from openai import OpenAI
import openai
import yaml
import re
import time
from typing import List, Optional # Added Optional for Python < 3.10 compatibility
from grader import evaluate_manifest # Assuming grader.py is in the same directory or PYTHONPATH

def grade_function(original_manifest, model_response):
    """Grade the model's response using the custom grader."""
    from grader import evaluate_manifest # Assuming grader.py is in the same directory or accessible.
    
    try:
        # Extract YAML from response (handle markdown formatting)
        fixed_yaml = extract_yaml(model_response)
        
        # Evaluate using our grader
        result = evaluate_manifest(original_manifest, fixed_yaml)
        
        return {
            "reward": result["reward"],
            "metadata": {
                "original_criticals": result.get("original_criticals", 0),
                "remaining_criticals": result.get("remaining_criticals", 0),
                "newly_introduced_criticals": result.get("newly_introduced_criticals", 0),
                "reason": result.get("reason", "No reason provided")
            }
        }
    except Exception as e:
        print(f"Error in grading: {e}")
        return {"reward": 0.0, "metadata": {"error": str(e)}, "newly_introduced_criticals": "N/A", "reason": "Processing error"}

def extract_yaml(response):
    """Extract YAML from model response that might contain markdown."""
    # Look for ```yaml blocks
    if "```yaml" in response:
        parts = response.split("```yaml")
        if len(parts) > 1:
            yaml_part = parts[1].split("```")[0].strip()
            return yaml_part
    
    # Look for ```yml blocks
    if "```yml" in response:
        parts = response.split("```yml")
        if len(parts) > 1:
            yaml_part = parts[1].split("```")[0].strip()
            return yaml_part
            
    # If no code blocks, assume the whole response is YAML
    return response.strip()

def evaluate_o4_mini_on_batch():
    """Loads problematic manifests, gets responses from o4-mini, and grades them."""
    try:
        with open("generated_manifests_output/problematic_dataset.json", 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: 'generated_manifests_output/problematic_dataset.json' not found.")
        print("Please ensure the dataset exists at the specified path.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from 'generated_manifests_output/problematic_dataset.json'.")
        return

    # Limit to 20 examples for micro-validation
    examples_to_evaluate = dataset[:20]
    
    if not examples_to_evaluate:
        print("No examples found in the dataset to evaluate.")
        return

    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except KeyError:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    print(f"Evaluating o4-mini on {len(examples_to_evaluate)} examples...")
    all_grades = []

    for i, item in enumerate(examples_to_evaluate):
        original_manifest = item.get("problematic_yaml")
        if original_manifest is None: # Check for None explicitly
            print(f"Skipping example {i+1} due to missing 'problematic_yaml' key or null value.")
            all_grades.append({"reward": 0.0, "metadata": {"error": "Missing problematic_yaml"}, "newly_introduced_criticals": "N/A", "reason": "Processing error"})
            continue

        print(f"\nProcessing example {i+1}/{len(examples_to_evaluate)}...")
        try:
            print("  Getting response from o4-mini...")
            api_response = client.responses.create(
                model="o4-mini",
                # reasoning={"effort": "medium"}, # Default is medium, can be added if specific effort is needed
                input=[{"role": "user", "content": original_manifest}],
                max_output_tokens=2000 # As per original script's intent for output length constraint
            )
            
            model_output_text = api_response.output_text
            print(f"  o4-mini response status: {api_response.status}")

            if api_response.status == "incomplete":
                 print(f"  Incomplete reason: {api_response.incomplete_details.reason if api_response.incomplete_details else 'Unknown'}")
            
            # Grade even if output is empty or incomplete, grade_function should handle it
            print("  Grading response...")
            grade_result = grade_function(original_manifest, model_output_text if model_output_text is not None else "")
            all_grades.append(grade_result)
            # Enhanced logging for grade details
            reward = grade_result.get('reward', 'N/A')
            metadata = grade_result.get('metadata', {})
            original_crit = metadata.get('original_criticals', 'N/A')
            remaining_crit = metadata.get('remaining_criticals', 'N/A')
            # The evaluate_manifest now returns newly_introduced_criticals and reason directly in its dict
            newly_introduced_crit = grade_result.get('newly_introduced_criticals', 'N/A') 
            reason_str = grade_result.get('reason', 'No reason provided')
            print(f"  Grade for example {i+1}: Reward: {reward}, OrigCrit: {original_crit}, RemCrit: {remaining_crit}, NewCrit: {newly_introduced_crit}")
            print(f"  Reason: {reason_str}")

        except Exception as e:
            print(f"  Error processing example {i+1} with OpenAI API or during grading: {e}")
            all_grades.append({"reward": 0.0, "metadata": {"error": f"Runtime error: {str(e)}"}, "newly_introduced_criticals": "N/A", "reason": "Processing error"})
            
    print("\n--- Evaluation Summary ---")
    if not all_grades: # Should not happen if examples_to_evaluate was not empty
        print("No examples were graded.")
        return
        
    successful_grades = [g for g in all_grades if 'error' not in g.get('metadata', {}) and g.get('reward') is not None]
    
    if successful_grades:
        total_reward = sum(g.get('reward', 0) for g in successful_grades)
        avg_reward = total_reward / len(successful_grades)
        print(f"Average Reward over {len(successful_grades)} successfully graded examples: {avg_reward:.4f}")
    else:
        print("No examples were successfully graded.")

    num_processing_errors = len(all_grades) - len(successful_grades)
    print(f"Number of examples with processing/grading errors: {num_processing_errors}")

# --- Configuration ---
# Ensure OPENAI_API_KEY is set in your environment
# For example: export OPENAI_API_KEY='your_key_here'
# The grader.py script also uses openai.OpenAI(), so the key should be globally available.
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client once for the script
# The grader.py will initialize its own client instance if it also uses one.
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set correctly.")
    # Depending on desired behavior, you might want to exit here:
    # import sys
    # sys.exit(1)
    client = None # Allow script to continue if user wants to see flow without API calls (e.g. manifest loading)

# User-specified model for generation.
# Set this to "o4-mini" to use the reasoning model for generation.
GENERATION_MODEL = "o4-mini" # Changed to o4-mini as per user's focus
FINETUNE_BASE_MODEL = "o4-mini" # As requested by user for fine-tuning target

FINETUNE_MODEL_SUFFIX = "k8s-rft-fixer-v1"
# Adjusted based on documentation for reasoning models; includes reasoning + output tokens.
# User should monitor usage and adjust. Start with a more generous value than typical chat models.
# The user's existing code snippet in rft_trainer.py used 2000 for o4-mini max_output_tokens.
# Let's use 4000 as a starting point for the micro-batch.
MAX_TOKENS_GENERATION = 4000
NUM_EXAMPLES_MICRO_RUN = 20
NUM_EPOCHS_MICRO_RUN = 10 # For (20 examples * 10 epochs = 200 steps)

PROBLEMATIC_MANIFESTS_FILE = "generated_manifests_output/problematic_dataset.json"
FINETUNING_DATA_FILE = "finetuning_data_for_o4mini.jsonl"

# --- Helper Functions ---

def load_problematic_manifests(json_file_path: str, num_to_load: int) -> List[str]:
    """Loads problematic manifest strings from the specified JSON dataset file."""
    manifests = []
    try:
        with open(json_file_path, 'r') as f:
            dataset = json.load(f)
        
        for item in dataset:
            if 'problematic_yaml' in item and item['problematic_yaml'] is not None: # Ensure YAML is not null
                manifests.append(item['problematic_yaml'])
            if len(manifests) >= num_to_load:
                break
        if not manifests:
            print(f"Warning: No valid 'problematic_yaml' entries found in the first {len(dataset)} items of {json_file_path}, or dataset is empty.")
        elif len(manifests) < num_to_load:
            print(f"Warning: Loaded only {len(manifests)} manifests, requested {num_to_load}.")

    except FileNotFoundError:
        print(f"Error: Manifests file not found at {json_file_path}")
        return [] # Return empty list on error
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return [] # Return empty list on error
    except Exception as e:
        print(f"An unexpected error occurred loading manifests: {e}")
        return [] # Return empty list on error
    return manifests


def extract_yaml_from_markdown(markdown_text: str) -> Optional[str]:
    """Extracts YAML content from a markdown code block, with improved handling for raw YAML possibly starting with a fence."""
    if not markdown_text:
        return None

    stripped_text = markdown_text.strip()

    # Priority 1: Standard fenced code blocks
    # ```yaml ... ```
    match_yaml = re.search(r"```yaml\n(.*?)\n```", stripped_text, re.DOTALL)
    if match_yaml:
        return match_yaml.group(1).strip()

    # ```yml ... ```
    match_yml = re.search(r"```yml\n(.*?)\n```", stripped_text, re.DOTALL)
    if match_yml:
        return match_yml.group(1).strip()
    
    # Generic ``` ... ```
    match_generic = re.search(r"```\n(.*?)\n```", stripped_text, re.DOTALL)
    if match_generic:
        content = match_generic.group(1).strip()
        # If the content of generic block itself starts with 'yaml\n', common if model does ```\nyaml\n...```
        if content.startswith("yaml\n"): 
            content = content.split('\\n', 1)[1] if '\\n' in content else content # Remove "yaml" line
        return content.strip()

    # Priority 2: Handle cases where the response might be *only* the content of a code block,
    # potentially with the opening fence but not the closing one, or vice-versa, or just raw YAML.

    # Case: Text starts with ```yaml\n but wasn't caught by the full block regex above (e.g., missing closing ```)
    if stripped_text.startswith("```yaml\n"):
        print("Warning: Text starts with ```yaml but not as a full block. Attempting to extract content.")
        # Remove the first line (```yaml) and treat the rest as potential YAML
        return stripped_text.split('\\n', 1)[1].strip() if '\\n' in stripped_text else ""

    # Case: Text starts with ```\n (generic fence) but not a full block
    if stripped_text.startswith("```\n"):
        print("Warning: Text starts with ``` but not as a full block. Attempting to extract content.")
        content_after_fence = stripped_text.split('\\n', 1)[1].strip() if '\\n' in stripped_text else ""
        # If this content itself starts with 'yaml\n' (e.g. ``` \n yaml \n ... )
        if content_after_fence.startswith("yaml\n"):
            content_after_fence = content_after_fence.split('\\n', 1)[1] if '\\n' in content_after_fence else ""
        return content_after_fence.strip()
        
    # Priority 3: If no fences detected at all, check if it looks like YAML directly
    # This was the part that triggered the warning before.
    # Now, it's reached only if no fences were found at the start by the logic above.
    if ":" in stripped_text and ("apiVersion:" in stripped_text or "kind:" in stripped_text or "metadata:" in stripped_text):
        print("Warning: No markdown block structure detected. Assuming the entire response is YAML.")
        return stripped_text
    
    print(f"Warning: Could not reliably extract YAML from response snippet:\\n{markdown_text[:300]}...")
    return None


def generate_fixed_manifest_openai(problematic_manifest: str, openai_client: OpenAI) -> Optional[str]:
    """Generates a fixed manifest using OpenAI's Responses API (for reasoning models like o4-mini)."""
    if not openai_client:
        print("Error: OpenAI client not initialized. Cannot generate manifest.")
        return None

    system_prompt = "You are an expert Kubernetes security engineer. Your task is to fix the provided problematic Kubernetes manifest. Only output the corrected YAML manifest, enclosed in ```yaml ... ``` markdown tags. Do not add any explanations before or after the YAML block."
    user_prompt_content = f"Problematic Kubernetes Manifest:\\n```yaml\\n{problematic_manifest}\\n```\\n\\nCorrected Kubernetes Manifest:"

    try:
        print(f"  Calling OpenAI model ({GENERATION_MODEL}) with max_output_tokens={MAX_TOKENS_GENERATION}...")
        response = openai_client.responses.create(
            model=GENERATION_MODEL, # Should be "o4-mini"
            input=[
                {"role": "system", "content": system_prompt}, # System message might be ignored or handled differently by `responses.create`
                {"role": "user", "content": user_prompt_content}
            ],
            reasoning={"effort": "medium"}, # Default, can be "low" or "high"
            max_output_tokens=MAX_TOKENS_GENERATION
        )
        
        print(f"  Response status: {response.status}")
        if response.usage:
            print(f"  Token usage: Input={response.usage.input_tokens}, Output={response.usage.output_tokens} (Reasoning: {response.usage.output_tokens_details.reasoning_tokens if response.usage.output_tokens_details else 'N/A'}), Total={response.usage.total_tokens}")

        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "Unknown"
            print(f"  Warning: OpenAI response was incomplete. Reason: {reason}")
            if reason == "max_output_tokens" and not response.output_text:
                print("  Critical: Ran out of tokens during reasoning, before generating any visible output. Consider increasing MAX_TOKENS_GENERATION.")
            elif response.output_text:
                 print("  Partial output was generated despite being incomplete.")
            # Continue to attempt extraction if there's any output_text
        
        if not response.output_text:
            print("  Error: OpenAI API returned no output_text.")
            return None
            
        return extract_yaml_from_markdown(response.output_text)

    except openai.APIError as e: # More specific error catching
        print(f"  Error calling OpenAI API (Responses API): {e}")
        if hasattr(e, 'response') and e.response and e.response.text:
            try:
                error_detail = json.loads(e.response.text)
                print(f"  API Error details: {error_detail}")
            except json.JSONDecodeError:
                print(f"  API Error details (raw): {e.response.text}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during OpenAI API call: {type(e).__name__} - {e}")
        return None

# --- Main Logic ---

def main():
    if not client:
        print("❌ Error: OpenAI client failed to initialize at the start of the script. Exiting.")
        return

    print("=======================================================")
    print("       🚀 RFT Micro-Validation Run Started 🚀")
    print("=======================================================")
    print(f"🧬 Using Generation Model: {GENERATION_MODEL}")
    print(f"🎯 Target Fine-Tune Base Model: {FINETUNE_BASE_MODEL}")
    print(f"⚖️ Grading with: grader.evaluate_manifest")
    print(f"⚙️ Max Examples: {NUM_EXAMPLES_MICRO_RUN}, Max Tokens (Generation): {MAX_TOKENS_GENERATION}")
    print("-------------------------------------------------------")

    # 1. Load problematic manifests
    print("📚 Loading problematic manifests...")
    problematic_manifests = load_problematic_manifests(PROBLEMATIC_MANIFESTS_FILE, NUM_EXAMPLES_MICRO_RUN)
    if not problematic_manifests:
        print(f"❌ Error: No problematic manifests loaded from {PROBLEMATIC_MANIFESTS_FILE}. Exiting.")
        return
    
    actual_examples_to_process = min(len(problematic_manifests), NUM_EXAMPLES_MICRO_RUN)
    print(f"✅ Loaded {len(problematic_manifests)} manifests. Will process {actual_examples_to_process} for this micro-run.")

    # 2. Generate fixes, grade, and prepare fine-tuning data
    fine_tuning_data_for_openai = []
    all_scores = []
    successful_generations_and_grades = 0
    api_call_errors = 0
    grading_errors = 0

    for i in range(actual_examples_to_process):
        original_manifest_str = problematic_manifests[i]
        print(f"\n----------------------------------------")
        print(f"✨ Processing Manifest [ {i+1} / {actual_examples_to_process} ] ✨")
        print(f"----------------------------------------")

        corrected_yaml_str = generate_fixed_manifest_openai(original_manifest_str, client)

        if corrected_yaml_str:
            print(f"  ✅ YAML successfully generated and extracted.")
            try:
                grading_result = evaluate_manifest(original_yaml=original_manifest_str, fixed_yaml=corrected_yaml_str)
                
                score = grading_result.get("reward", 0.0)
                reason = grading_result.get("reason", "No reason provided by grader.")
                all_scores.append(score)
                
                print(f"  📜 Grading Details:")
                print(f"    Original Criticals: {grading_result.get('original_criticals', 'N/A')}")
                print(f"    Remaining Criticals: {grading_result.get('remaining_criticals', 'N/A')}")
                print(f"    New Criticals: {grading_result.get('newly_introduced_criticals', 'N/A')}")
                print(f"    🏅 Graded Score: {score:.4f}")
                print(f"    🗣️ Grader Reason: {reason}")

                if grading_result.get("is_valid_yaml", False):
                    successful_generations_and_grades += 1
                    system_message_ft = "You are a Kubernetes manifest correction assistant. Given a problematic manifest, provide the corrected version as a YAML code block."
                    user_message_ft = f"Problematic Kubernetes Manifest:\\n```yaml\\n{original_manifest_str}\\n```\\n\\nCorrected Kubernetes Manifest:"
                    assistant_message_ft = f"```yaml\\n{corrected_yaml_str}\\n```"
                    fine_tuning_data_for_openai.append({
                        "messages": [
                            {"role": "system", "content": system_message_ft},
                            {"role": "user", "content": user_message_ft},
                            {"role": "assistant", "content": assistant_message_ft}
                        ]
                    })
                else:
                    print(f"  ⚠️ Grader Warning: Generated YAML was considered invalid by the grader - Reason: {grading_result.get('reason', 'Unknown')}")

            except Exception as e:
                grading_errors += 1
                print(f"  ❌ Error during grading with evaluate_manifest: {e}")
                all_scores.append(0.0) # Penalize grading errors
        else:
            api_call_errors +=1
            print("  ❌ Error: Failed to generate or extract corrected manifest from OpenAI.")
            all_scores.append(0.0) # Penalize generation failure
        
        if i < actual_examples_to_process - 1: # Avoid printing pause after the last item
             print(f"  ⏱️ Pausing for 20s (API rate limit)...")
        time.sleep(20)

    # Output summary of scores
    print("\n=======================================================")
    print("        📊 RFT Micro-Validation Summary 📊")
    print("=======================================================")
    
    total_processed = len(all_scores)
    print(f"Total Manifests Processed: {total_processed}")
    print(f"✅ Successfully Generated & Graded (Valid YAML): {successful_generations_and_grades}")
    print(f"📉 OpenAI API Call/Extraction Failures: {api_call_errors}")
    print(f"📈 Grading Logic Errors: {grading_errors}")

    if total_processed > 0:
        avg_score_overall = sum(all_scores) / total_processed if total_processed > 0 else 0.0
        print(f"⭐ Average Score (all processed, including failures): {avg_score_overall:.4f}")
        
        # Calculate average score for only successfully generated and graded items
        # This requires scores from only the `successful_generations_and_grades` count
        # We can sum the scores of the items that were added to `fine_tuning_data_for_openai`
        # Or, more simply, filter `all_scores` based on successful processing if we had a flag per item.
        # For now, let's report average of scores > 0 for a proxy of "successful attempts that got some reward"
        positive_scores = [s for s in all_scores if s > 0]
        if positive_scores:
            avg_positive_score = sum(positive_scores) / len(positive_scores)
            print(f"🎯 Average Score (for attempts with reward > 0): {avg_positive_score:.4f} (based on {len(positive_scores)} attempts)")
        else:
            print("🎯 No attempts resulted in a reward > 0.")
    else:
        print("No manifests were processed or no scores were recorded.")
    
    print("\n💡 This summary helps confirm if the reward signal from grader.py is working as expected.")
    print("-------------------------------------------------------")

    # 3. Prepare and provide instructions for fine-tuning job
    if fine_tuning_data_for_openai:
        print(f"\n📦 Prepared {len(fine_tuning_data_for_openai)} examples for potential fine-tuning.")
        
        with open(FINETUNING_DATA_FILE, 'w') as f:
            for item in fine_tuning_data_for_openai:
                f.write(json.dumps(item) + "\n")
        print(f"💾 Fine-tuning data (OpenAI chat format) saved to: {FINETUNING_DATA_FILE}")

        print(f"\n📋 --- Fine-Tuning Instructions (Verify Model Compatibility) ---")
        print(f"❗ IMPORTANT: The model '{FINETUNE_BASE_MODEL}' is a reasoning model. ")
        print(f"   Standard fine-tuning (`openai api fine_tunes.create`) may NOT be supported or may have a different procedure.")
        print(f"   👉 PLEASE VERIFY OpenAI's official documentation for fine-tuning '{FINETUNE_BASE_MODEL}'.")
        print(f"   The following are standard instructions for models like GPT-3.5-turbo:")

        print(f"   1. Install/Upgrade OpenAI CLI: `pip install --upgrade openai`")
        print(f"   2. Upload data file: `openai api files.create --purpose fine-tune --file {FINETUNING_DATA_FILE}` (outputs FILE_ID)")
        print(f"   3. Create fine-tuning job (IF '{FINETUNE_BASE_MODEL}' supports this, replace FILE_ID):")
        print(f"      `openai api fine_tunes.create --training_file FILE_ID --model {FINETUNE_BASE_MODEL} --suffix \"{FINETUNE_MODEL_SUFFIX}\" --n_epochs {NUM_EPOCHS_MICRO_RUN}`")
        training_steps_info = actual_examples_to_process * NUM_EPOCHS_MICRO_RUN if actual_examples_to_process > 0 else 'N/A'
        print(f"      (Epochs aim for ~{training_steps_info} training steps for {actual_examples_to_process} examples)")
        
        print("\n🔍 Review scores and generated .jsonl data carefully before any actual training job.")
    else:
        print("\n🤷 No data prepared for fine-tuning (no successful generation & grading of valid YAMLs, or issues encountered).")
    print("=======================================================")
    print("                 ✨ Run Complete ✨")
    print("=======================================================")

if __name__ == "__main__":
    main() 