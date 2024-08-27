import streamlit as st
import dspy
import pandas as pd
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO
from typing import List, Dict
import os

# Set up available LLMs
LLM_OPTIONS = {
    "GPT-3.5-turbo": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
    "GPT-4-turbo": "gpt-4-1106-preview",
}

class PromptOptimizer(dspy.Signature):
    """Optimize the given prompt based on the workflow."""
    input_prompt = dspy.InputField()
    workflow = dspy.InputField()
    optimized_prompt = dspy.OutputField()

class OptimizePrompt(dspy.Module):
    def __init__(self):
        super().__init__()
        self.optimize = dspy.ChainOfThought(PromptOptimizer)

    def forward(self, input_prompt, workflow):
        return self.optimize(input_prompt=input_prompt, workflow=workflow)

class PromptJudge(dspy.Signature):
    """Judge if the optimized prompt is better than the original."""
    original_prompt = dspy.InputField()
    optimized_prompt = dspy.InputField()
    workflow = dspy.InputField()
    is_better = dspy.OutputField(desc="Is the optimized prompt better?", prefix="Better[Yes/No]:")

judge = dspy.ChainOfThought(PromptJudge)

def prompt_quality_metric(example, pred):
    is_better = judge(
        original_prompt=example.input_prompt,
        optimized_prompt=pred.optimized_prompt,
        workflow=example.workflow
    )
    return int(is_better.is_better == "Yes")

def create_dataset(prompts: List[str], workflows: List[str]) -> List[dspy.Example]:
    return [dspy.Example(input_prompt=p, workflow=w, optimized_prompt=p) for p, w in zip(prompts, workflows)]

def optimize_prompts(prompts: List[str], workflows: List[str], teleprompter, lm) -> List[str]:
    optimizer = OptimizePrompt()
    dataset = create_dataset(prompts, workflows)
    
    if isinstance(teleprompter, COPRO):
        compiled_optimizer = teleprompter.compile(
            optimizer,
            trainset=dataset,
            eval_kwargs=dict(metric=prompt_quality_metric)
        )
    elif isinstance(teleprompter, BootstrapFewShot):
        compiled_optimizer = teleprompter.compile(
            optimizer,
            trainset=dataset
        )
    elif isinstance(teleprompter, BootstrapFewShotWithRandomSearch):
        compiled_optimizer = teleprompter.compile(
            optimizer,
            trainset=dataset,
            metric=prompt_quality_metric
        )
    else:
        raise ValueError(f"Unsupported teleprompter type: {type(teleprompter)}")
    
    optimized_prompts = []
    for prompt, workflow in zip(prompts, workflows):
        result = compiled_optimizer(input_prompt=prompt, workflow=workflow)
        optimized_prompts.append(result.optimized_prompt)
    return optimized_prompts

def run_optimization(mode: str, input_data: str, workflow: str, teleprompter, lm) -> Dict[str, List[str]]:
    if mode == "Single prompt":
        original_prompts = [input_data]
        workflows = [workflow]
    else:  # Mass prompt
        df = pd.read_csv(input_data)
        original_prompts = df['prompt'].tolist()
        workflows = [workflow] * len(original_prompts)
    
    optimized_prompts = optimize_prompts(original_prompts, workflows, teleprompter, lm)
    return {"Original": original_prompts, "Optimized": optimized_prompts}

st.title("DSPy Prompt Optimizer")

# LLM selection
selected_llm = st.selectbox("Select Language Model:", list(LLM_OPTIONS.keys()))

# Set OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    lm = dspy.OpenAI(model=LLM_OPTIONS[selected_llm])
    dspy.settings.configure(lm=lm)
else:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

# Set up available teleprompters
TELEPROMPTER_OPTIONS = {
    "BootstrapFewShot": lambda: BootstrapFewShot(max_bootstrapped_demos=3, max_labeled_demos=5),
    "BootstrapFewShotWithRandomSearch": lambda: BootstrapFewShotWithRandomSearch(metric=prompt_quality_metric, max_bootstrapped_demos=2, num_candidate_programs=3),
    "COPRO": lambda: COPRO(prompt_model=dspy.OpenAI(model="gpt-3.5-turbo"), metric=prompt_quality_metric)
}

# Teleprompter selection
selected_teleprompter = st.selectbox("Select Teleprompter:", list(TELEPROMPTER_OPTIONS.keys()))
teleprompter = TELEPROMPTER_OPTIONS[selected_teleprompter]()

mode = st.radio("Select mode:", ["Single prompt", "Mass prompt"])

if mode == "Single prompt":
    input_prompt = st.text_area("Enter your prompt:")
    file_upload = None
else:
    file_upload = st.file_uploader("Upload CSV file (with 'prompt' column):", type=['csv'])
    input_prompt = None

workflow = st.text_area("Enter the workflow for prompt optimization:")

if st.button("Optimize"):
    if (mode == "Single prompt" and input_prompt) or (mode == "Mass prompt" and file_upload):
        with st.spinner("Optimizing prompts..."):
            input_data = input_prompt if mode == "Single prompt" else file_upload
            results = run_optimization(mode, input_data, workflow, teleprompter, lm)
            
            st.subheader("Results:")
            for original, optimized in zip(results["Original"], results["Optimized"]):
                st.write("Original prompt:", original)
                st.write("Optimized prompt:", optimized)
                st.write("---")
            
            # Create a DataFrame for download
            df_results = pd.DataFrame(results)
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="optimized_prompts.csv",
                mime="text/csv",
            )
    else:
        st.error("Please provide the required input based on the selected mode.")
