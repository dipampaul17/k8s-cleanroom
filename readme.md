# k8s manifest rft experimentation

![python version](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge)
![openai](https://img.shields.io/badge/openai-API-green?style=for-the-badge)
![kube-linter](https://img.shields.io/badge/kube--linter-enabled-brightgreen?style=for-the-badge)
![reasoning model](https://img.shields.io/badge/model-o4--mini-purple?style=for-the-badge)
![status](https://img.shields.io/badge/status-experimental-orange?style=for-the-badge)
![license](https://img.shields.io/badge/license-unlicensed-lightgrey?style=for-the-badge)

## purpose

this project is an exploration into using reinforcement learning from fine-tuning (rft) with openai's advanced reasoning models (specifically `o4-mini`) to automatically identify and correct security misconfigurations and other issues in kubernetes manifests. the primary goal is to see if a reward signal, derived from a static analysis tool (`kube-linter`), can guide a language model to become better at producing secure and valid kubernetes yaml.

## core idea & intuition

the core intuition is that while large language models have impressive generative capabilities, guiding them effectively for specific, structured tasks like code correction requires a feedback mechanism.

*   **reinforcement learning (rl) principles**: instead of just supervised fine-tuning on "correct" examples, we want the model to learn from a scalar reward that signifies how "good" its proposed fix is. this allows the model to explore and learn from its mistakes and successes.

*   **`kube-linter` as the oracle**: `kube-linter` is a static analysis tool that checks kubernetes yaml files for common misconfigurations and adherence to best practices. we use its output to quantify the "goodness" of a manifest, forming the basis of our reward signal. a fix that reduces critical `kube-linter` issues (without introducing new ones) gets a higher reward.

*   **reasoning models (`o4-mini`)**: models like `o4-mini` are designed for complex problem-solving and multi-step reasoning. the hypothesis is that these capabilities are well-suited for understanding the structure and intent of a kubernetes manifest and applying targeted corrections, rather than just superficial textual changes.

*   **iterative improvement**: the long-term vision of rft is an iterative loop where the model generates fixes, receives rewards, is fine-tuned based on this feedback, and then generates even better fixes. this project sets up the initial stage: generating rewarded data.

## why this approach matters

kubernetes manifests present unique challenges for ai correction:

1. **structural constraints**: valid yaml with specific kubernetes schema requirements
2. **semantic understanding**: grasping what each field does and how they relate to security
3. **security implications**: some changes might fix one issue but introduce others
4. **balance**: maintaining functionality while addressing security concerns

traditional approaches might:
- use rule-based systems (limited adaptability)
- train on paired examples (requires large curated datasets)
- use general llms (might not understand k8s-specific security nuances)

our rft approach aims to:
- learn from feedback, not just examples
- generalize to new types of issues not seen during training
- prioritize critical security fixes without breaking functionality
- leverage reasoning capabilities to understand manifest structure holistically

## workflow visualization

```
┌──────────────────┐     ┌────────────────┐     ┌────────────────┐
│  problematic     │     │                │     │    fixed       │
│    manifest      │────▶│   o4-mini      │────▶│   manifest     │
│                  │     │                │     │                │
└──────────────────┘     └────────────────┘     └────────┬───────┘
                                                         │
                                                         ▼
┌──────────────────┐     ┌────────────────┐     ┌────────────────┐
│  fine-tuning     │     │                │     │                │
│     data         │◀────│   reward       │◀────│   kube-linter  │
│                  │     │                │     │    grading     │
└──────────────────┘     └────────────────┘     └────────────────┘
```

## how it works (current implementation)

the current setup focuses on generating a dataset of (problematic_manifest, model_generated_fix, reward) tuples, which could then theoretically be used for fine-tuning.

1.  **load problematic data**: `rft_trainer.py` starts by loading a set of known "problematic" kubernetes manifests from `generated_manifests_output/problematic_dataset.json`.

2.  **generate fixes**: for each problematic manifest, it prompts the `o4-mini` model (using `client.responses.create` for reasoning) to provide a corrected version. the prompt explicitly asks for the output to be a yaml code block.

3.  **extract yaml**: the script then carefully extracts the yaml content from the model's potentially markdown-formatted response.

4.  **grade the fix**: the `grader.py` script takes both the original problematic manifest and the model's proposed fix.
    *   it runs `kube-linter` (configured via `.kube-linter.yaml`) on both manifests.
    *   it calculates a reward score based on:
        *   reduction in critical issues.
        *   crucially, a heavy penalty for any *new* critical issues introduced by the fix.
        *   perfect fixes (all original issues gone, no new ones) get the max reward (1.0). clean original manifests that remain clean also get 1.0.
    *   `grader.py` also uses `o4-mini` to generate a concise, human-readable assessment of the changes.

5.  **log and prepare data**: `rft_trainer.py` logs the process, scores, and then formats the successful attempts (original manifest, model's fix) into a `.jsonl` file (`finetuning_data_for_o4mini.jsonl`) suitable for openai's fine-tuning api (chat completions format).

## design choices & implementation details

several important design decisions shape the current implementation:

### reward function design

the reward function (in `grader.py`) is carefully calibrated with these principles:

```python
# Simplified reward calculation logic
if actual_original_critical_count == 0:
    if newly_introduced_criticals_count_py > 0:
        reward = 0.0  # Penalize introducing issues to a clean manifest
    else:
        reward = 1.0  # Manifest was clean and remains clean
else:  # actual_original_critical_count > 0
    if actual_fixed_critical_count == 0 and newly_introduced_criticals_count_py == 0:
        reward = 1.0  # All issues fixed, no new ones
    else:
        # Consider new issue types as negating some of the fixes
        improvement_score = (actual_original_critical_count - actual_fixed_critical_count - newly_introduced_criticals_count_py) / actual_original_critical_count
        reward = max(0.0, min(0.8, improvement_score))  # Cap partial fixes at 0.8
```

key design principles:
- zero tolerance for new issues in previously clean manifests (reward = 0)
- perfect fixes are highly rewarded (reward = 1.0)
- partial improvements get proportional rewards (up to 0.8)
- introducing new issues is heavily penalized
- the reward is normalized based on the original issue count

### model prompt engineering

the system prompt used for `o4-mini` is intentionally simple to establish a baseline:

```
"You are an expert Kubernetes security engineer. 
Your task is to fix the provided problematic Kubernetes manifest. 
Only output the corrected YAML manifest, enclosed in ```yaml ... ``` markdown tags. 
Do not add any explanations before or after the YAML block."
```

this:
- sets the model in an expert role
- constrains the output format
- avoids explanation that could confuse the yaml extraction

### yaml extraction robustness

extracting yaml from model responses is trickier than it sounds. our function handles multiple cases:
- standard ```yaml ... ``` blocks
- unmarked code blocks that contain yaml
- responses with partial markdown formatting
- direct yaml without any markdown

### cli design for developer experience

the `rft_trainer.py` script's output is designed to be:
- process-transparent: showing each step clearly
- visually distinguishable: using characters like ✨, ✅, and ❌
- progress-indicating: showing current manifest being processed
- quantitative: providing clear metrics about reward and issue counts
- summarized: giving aggregated statistics at the end

## observed results

from initial testing with 20 problematic manifests:

- **average reward score**: ~0.22 (on a scale of 0.0 to 1.0)
- **success rate**: 90% of attempts (18/20) yielded some improvement
- **zero-reward cases**: 2 manifests got a 0.0 reward due to introducing many new critical issues
- **best case**: achieved a 0.5 reward by fixing 15+ issues while introducing only 1 new issue
- **common fixes**: model consistently addressed issues like:
  - fixing `latest` image tags with specific versions
  - adding resource limits
  - addressing privilege escalation concerns
  - fixing user permissions (non-root)
  - enabling read-only root filesystems
- **common mistakes**: the model sometimes:
  - introduced new security issues when trying to fix existing ones
  - made syntax errors in complex manifests
  - fixed superficial issues but missed deeper security concerns

## key components

*   **`rft_trainer.py`**: the main orchestration script. it manages the data loading, interaction with the openai api for generating fixes, calls the grader, and prepares the fine-tuning dataset. its cli output is designed to be clean and informative.

*   **`grader.py`**: responsible for evaluating the quality of the fix. it uses `kube-linter` as its ground truth and implements the reward function. it also uses an llm for a qualitative summary.

*   **`.kube-linter.yaml`**: configuration file for `kube-linter`, specifying which checks are enabled or disabled. `addallbuiltin: true` is currently used to enable all standard checks.

*   **`generated_manifests_output/problematic_dataset.json`**: the input dataset of kubernetes manifests with known issues. (the generation of this file itself is outside the scope of `rft_trainer.py` but is a prerequisite).

*   **`finetuning_data_for_o4mini.jsonl`**: the output file containing prompt-completion pairs (in chat format) and associated metadata, ready for potential fine-tuning.

*   **`.env`**: used to store the `openai_api_key`.

## setup & usage

1.  **prerequisites**:
    *   python (3.9+ recommended).
    *   `pip` (python package installer).
    *   `kube-linter` installed and available in your system's path.
    *   an openai api key.

2.  **environment setup**:
    *   create a file named `.env` in the project root.
    *   add your openai api key to it: `openai_api_key='your_sk_xxxxxxxxxxxxxx_key_here'`

3.  **install dependencies**:
    ```bash
    pip install python-dotenv openai pyyaml
    ```

4.  **run the micro-validation**:
    ```bash
    python rft_trainer.py
    ```
    this will process a small batch of examples (currently 20) from `generated_manifests_output/problematic_dataset.json` and generate `finetuning_data_for_o4mini.jsonl`.

## interpreting micro-validation results

the `rft_trainer.py` script provides detailed output:

*   **per-manifest processing**: logs api calls, token usage, extracted yaml, and grading details (original vs. remaining criticals, new criticals, reward score, and grader's reason).

*   **summary statistics**:
    *   total manifests processed.
    *   number of successful generations & grades.
    *   api call/extraction failures.
    *   grading logic errors.
    *   average reward score (overall, and for attempts with reward > 0).

*   **fine-tuning data**: path to the generated `.jsonl` file and instructions/caveats for proceeding with fine-tuning.

a low average reward (e.g., ~0.2-0.3) indicates the model is making some improvements but isn't consistently producing perfect fixes. a score of 0 means the fix was detrimental or made no positive impact according to the reward function. a score of 1.0 is ideal.

## thought process & current status

this project is intentionally experimental. the initial phase documented here was about:

*   **feasibility of reward signal**: can `kube-linter` output be translated into a meaningful scalar reward? the current `grader.py` suggests yes, as it penalizes new issues and rewards fixes.

*   **baseline model performance**: how well does `o4-mini` perform on this task with a straightforward prompt, before any fine-tuning? results show it *can* make improvements but also *can* introduce significant new errors.

*   **data pipeline**: establishing a workflow to generate (prompt, response, reward) tuples.

*   **developer experience**: creating a clean and informative cli for tracking the process.

### key learnings so far

1. **reasoning models show promise**: `o4-mini` demonstrates meaningful improvements in ~90% of cases, suggesting it has some understanding of kubernetes security concepts.

2. **reward design is critical**: our initial reward function seems to correctly incentivize the right behaviors (fixing issues without introducing new ones).

3. **the balancing act**: manifest fixing involves tradeoffs between different security concerns; a simple scalar reward may not capture all nuances.

4. **prompting matters**: even without fine-tuning, better prompts could likely improve performance significantly.

5. **extraction challenges**: getting clean yaml out of model responses requires robust parsing logic.

6. **inconsistent performance**: the model's effectiveness varies significantly across different manifests and issue types.

the system is not trying to be "intelligent" about *how* to fix things beyond what the llm provides; it's about seeing if the reward mechanism can, over time (with actual rft), teach the llm to be better.

## future directions

this initial setup paves the way for several interesting explorations:

*   **actual fine-tuning**:
    *   researching and implementing the correct procedure for fine-tuning reasoning models like `o4-mini` (if it differs from standard fine-tuning of models like gpt-3.5-turbo and is supported for this type of objective).
    *   running fine-tuning jobs with the generated data and evaluating if the model's performance improves on a holdout set.
    *   exploring different fine-tuning strategies, such as:
      - using only high-reward examples to guide learning
      - including low-reward examples as "what not to do"
      - incorporating a mix to provide balance

*   **prompt engineering**:
    *   iterating on the system prompt given to `o4-mini` for generating fixes. more detailed instructions, few-shot examples, or different personas might yield better baseline results.
    *   refining the prompt used in `grader.py` for the llm-generated assessment reason.
    *   testing structured prompting with explicit instructions about common k8s security issues.
    *   exploring chain-of-thought prompting to guide the model through a systematic review process.

*   **reward shaping**:
    *   experimenting with different reward function formulations in `grader.py`. for example, varying penalties for different severities of issues (if `kube-linter` provides this) or adding bonuses for specific positive changes.
    *   implementing weighted rewards that prioritize critical security fixes over less important issues.
    *   multi-objective rewards that balance security improvements with maintaining functionality.
    *   exploring non-linear reward functions that might better capture the value of fixing certain issue combinations.

*   **dataset expansion & diversity**:
    *   using a much larger and more diverse set of problematic kubernetes manifests, covering a wider array of misconfigurations.
    *   synthesizing more challenging edge cases to push the model's capabilities.
    *   creating benchmark datasets for specific k8s resource types (pods, deployments, services, etc.)
    *   gathering real-world examples of vulnerable manifests (properly anonymized).

*   **in-depth error analysis**:
    *   systematically analyzing the types of errors `o4-mini` makes (both in its fixes and when it introduces new problems) to inform prompt engineering or reward shaping.
    *   categorizing error patterns to understand model strengths and weaknesses.
    *   creating targeted training examples to address common failure modes.

*   **automated rft loop**:
    *   building a more comprehensive rft loop where the model is fine-tuned, re-evaluated on new problematic manifests, data is collected, and the cycle repeats, potentially leading to continuous improvement.
    *   implementing automatic halting criteria based on convergence or performance plateaus.
    *   tracking performance across iterations to measure learning progress.

*   **exploring alternative models**:
    *   comparing `o4-mini`'s performance with other openai models (e.g., gpt-4 series if api access allows for this scale) or even open-source llms if they have suitable reasoning and instruction-following capabilities.
    *   testing specialized code-focused models vs general purpose models.
    *   investigating the tradeoff between model size, cost, and effectiveness for this task.

*   **tool integration**:
    *   could other static analysis tools or security scanners be integrated into the grading process for a more holistic reward signal?
    *   exploring multi-tool validation with tools like:
      - kubeaudit
      - kyverno
      - polaris
      - falco
    *   developing hybrid approaches that combine rule-based fixes with llm-generated solutions.

*   **practical applications**:
    *   building a cli tool or github action that integrates the trained model.
    *   creating a vscode extension for real-time kubernetes manifest security feedback.
    *   exploring integration with ci/cd pipelines as a pre-deployment check.
    *   investigating potential use in kubernetes admission controllers.

## disclaimer

this is an experimental project for learning and exploration. the code and methodologies are not intended for direct use in production environments without significant further development, testing, and validation. the effectiveness of rft for this specific task is still under investigation. always be cautious when dealing with automated code generation or correction, especially for infrastructure configurations. 