import modal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate  # classification + text metrics
import os
from modal import Mount

app = modal.App("llama-baseline")
test_file_path = "dataset_test.csv"  # Local path (in the same directory)
remote_test_file_path = "/root/dataset_test.csv"  # Path inside the container
output_dir = "."  # Current directory
local_mount = Mount.from_local_dir(output_dir, remote_path="/outputs")

image = (
    modal.Image.debian_slim()
    .apt_install("pkg-config", "cmake")   # <--- Add these
    .pip_install(
        "transformers",
        "accelerate>=0.26",
        "torch",
        "pandas",
        "evaluate",
        "scikit-learn",   
        "sentencepiece",  
        "rouge_score",    # Needed for ROUGE evaluation
        "nltk",           # Needed for BLEU, METEOR
        "sacrebleu",      # Better BLEU implementation
        "python-dateutil" 
    )
    .add_local_file(test_file_path, remote_path=remote_test_file_path)  
)

class BaselineTestDataset(Dataset):
    """
    Dataset class for genomic sequence data.
    
    Assumes CSV has columns:
    - 'Sequence'          (str) => DNA sequence
    - 'dataset'           (str) => e.g. 'ClinVar' or 'COSMIC'
    - 'Ground_Annotation' (str) => ground-truth annotation text
    """
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.sequence_lengths = {idx: len(row.get("Sequence", "")) for idx, row in df.iterrows()}
        lengths = list(self.sequence_lengths.values())
        print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        print(f"Number of sequences > 5000 chars: {sum(1 for l in lengths if l > 5000)}/{len(lengths)}")
        self.prompt_template = (
            "Given this DNA sequence:\n{sequence}\n"
            "Please provide a detailed annotation of this sequence. Describe "
            "what database it belongs to (ClinVar or COSMIC), its potential function, "
            "and any clinical significance if applicable."
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        sequence = row.get("Sequence", "")
        MAX_SEQUENCE_LENGTH = 5000
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH] + "... [truncated]"
        prompt_text = self.prompt_template.format(sequence=sequence)
        

        return prompt_text, idx



@app.function(image=image, gpu="A100", timeout=600, mounts=[local_mount])
def run_llama_on_test(
    test_file: str = "/root/dataset_test.csv",
    output_file: str = "baseline_predictions.csv",
):
    """
    Zero-shot baseline evaluation using Llama model:
    1) Loads LLaMA model
    2) Reads test CSV with columns [Sequence, dataset, Ground_Annotation]
    3) Generates annotation for each sequence
    4) Extracts predicted dataset from generated text
    5) Computes metrics: Accuracy, F1, ROUGE-L, BLEU, METEOR
    6) Saves results to output_file

    Usage:
      modal run llama_inference.py
    """
    MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    print(f"✅ Loaded baseline model: {MODEL_NAME}")
    print(f"Loading test dataset from {test_file}")
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found at {test_file}")
        print(f"Current directory contents: {os.listdir('/')}")
        print(f"Root directory contents: {os.listdir('/root')}")
        return "Error: Test file not found."
    
    df_test = pd.read_csv(test_file)
    print(f"Loaded test dataset with {len(df_test)} rows")
    print(f"Columns in dataset: {df_test.columns.tolist()}")
    
    # Create dataset
    dataset_test = BaselineTestDataset(df_test)
    loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

    df_test["Predicted_Annotation"] = ""
    df_test["Predicted_Dataset"] = ""


    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")

    class_preds = []
    class_refs = []
    text_preds = []
    text_refs = []


    def generate_annotation(prompt):
        """Generate annotation text using the model."""
        inputs = tokenizer(prompt, return_tensors="pt")
        input_token_count = inputs.input_ids.shape[1]
        
        if not hasattr(generate_annotation, "token_counts"):
            generate_annotation.token_counts = []
        generate_annotation.token_counts.append(input_token_count)
        
        # Print token count for every 10th sample or if tokens > 1000
        sample_idx = len(generate_annotation.token_counts) - 1
        if sample_idx % 10 == 0 or input_token_count > 1000:
            print(f"Sample {sample_idx} - Input tokens: {input_token_count}")
        
        inputs = inputs.to("cuda")
        
        # Generate with error handling
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,  # Generate new tokens rather than limiting total length
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7 
                )
            
            # Calculate and print output token count
            output_token_count = out.shape[1] - inputs.input_ids.shape[1]
            if sample_idx % 10 == 0 or input_token_count > 1000:
                print(f"Sample {sample_idx} - Generated tokens: {output_token_count}")
            
            full_response = tokenizer.decode(out[0], skip_special_tokens=True)
            
            if prompt in full_response:
                response = full_response.replace(prompt, "").strip()
            else:
                response = full_response.strip()
                
            return response
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory error with {input_token_count} input tokens.")
                if hasattr(generate_annotation, "token_counts") and generate_annotation.token_counts:
                    print(f"Token count statistics so far: min={min(generate_annotation.token_counts)}, "
                          f"max={max(generate_annotation.token_counts)}, "
                          f"avg={sum(generate_annotation.token_counts)/len(generate_annotation.token_counts):.1f}")
                return "Error: Unable to process this sequence due to its length."
            else:
                raise

    def extract_dataset_label(annotation_text):
        """Extract dataset label from annotation text."""
        txt_lower = annotation_text.lower()
        
        # Check for direct mentions
        if "clinvar" in txt_lower:
            return "ClinVar"
        elif "cosmic" in txt_lower:
            return "COSMIC"
            
        # Check for more complex patterns (advanced pattern matching)
        clinvar_patterns = ["clinical variant", "clinical significance", "pathogenic", "benign", "likely pathogenic"]
        cosmic_patterns = ["somatic mutation", "cancer", "tumor", "malignant", "oncogene"]
        
        clinvar_score = sum(1 for pattern in clinvar_patterns if pattern in txt_lower)
        cosmic_score = sum(1 for pattern in cosmic_patterns if pattern in txt_lower)
        
        if clinvar_score > cosmic_score:
            return "ClinVar"
        elif cosmic_score > clinvar_score:
            return "COSMIC"
        else:
            return "Unknown"


    print(f"Generating predictions for {len(loader)} samples...")
    
    for idx, (prompt_text, row_data) in enumerate(loader):
        # Extract prompt string and row index (batch_size=1)
        prompt_str = prompt_text[0]
        row_idx = row_data[0].item()  # Convert tensor to scalar
        
        # Print progress every 10 samples
        if idx % 10 == 0:
            print(f"Processing sample {idx}/{len(loader)}...")
        
        # Generate annotation
        try:
            annotation_pred = generate_annotation(prompt_str)
            dataset_pred = extract_dataset_label(annotation_pred)
            
            # Store predictions in DataFrame using the index
            df_test.at[row_idx, "Predicted_Annotation"] = annotation_pred
            df_test.at[row_idx, "Predicted_Dataset"] = dataset_pred
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Store error as prediction for debugging
            df_test.at[row_idx, "Predicted_Annotation"] = f"ERROR: {str(e)}"
            df_test.at[row_idx, "Predicted_Dataset"] = "Unknown"

    if hasattr(generate_annotation, "token_counts") and generate_annotation.token_counts:
        token_counts = generate_annotation.token_counts
        print("\n=== Token Count Statistics ===")
        print(f"Min tokens: {min(token_counts)}")
        print(f"Max tokens: {max(token_counts)}")
        print(f"Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
        print(f"Samples with >1000 tokens: {sum(1 for t in token_counts if t > 1000)}/{len(token_counts)}")
        print(f"Samples with >2000 tokens: {sum(1 for t in token_counts if t > 2000)}/{len(token_counts)}")
        print(f"Samples with >5000 tokens: {sum(1 for t in token_counts if t > 5000)}/{len(token_counts)}")
    

    print("Preparing data for metrics calculation...")
    
    # Check if required columns exist
    if "dataset" in df_test.columns:
        class_preds = df_test["Predicted_Dataset"].tolist()
        class_refs = df_test["dataset"].tolist()
    else:
        print("Warning: No 'dataset' column found. Skipping classification metrics.")
        class_preds, class_refs = [], []

    if "Ground_Annotation" in df_test.columns:
        text_preds = df_test["Predicted_Annotation"].tolist()
        text_refs = df_test["Ground_Annotation"].tolist()
    else:
        print("Warning: No 'Ground_Annotation' column found. Skipping text metrics.")
        text_preds, text_refs = [], []


    print("Computing classification metrics...")
    metrics_results = {}
    
    if class_preds and class_refs:
        # Filter out "Unknown" predictions for proper metric calculation
        valid_indices = [i for i, pred in enumerate(class_preds) if pred != "Unknown"]
        filtered_preds = [class_preds[i] for i in valid_indices]
        filtered_refs = [class_refs[i] for i in valid_indices]
        
        if filtered_preds:
            # Compute accuracy
            correct = sum(1 for p, r in zip(filtered_preds, filtered_refs) if p == r)
            accuracy = correct / len(filtered_preds)
            print(f"Classification Accuracy: {accuracy:.4f}")
            metrics_results["accuracy"] = accuracy
            
            # Compute F1 score manually
            from sklearn.metrics import f1_score
            try:
                f1 = f1_score(filtered_refs, filtered_preds, average="macro", pos_label=None)
                print(f"Classification F1 (macro): {f1:.4f}")
                metrics_results["f1"] = f1
            except Exception as e:
                print(f"Error calculating F1 score: {e}")
                # Fallback to manual calculation if needed
                try:
                    # Convert string labels to integer indices
                    unique_labels = sorted(set(filtered_refs + filtered_preds))
                    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
                    idx_refs = [label_to_idx[label] for label in filtered_refs]
                    idx_preds = [label_to_idx[label] for label in filtered_preds]
                    
                    f1 = f1_score(idx_refs, idx_preds, average="macro")
                    print(f"Classification F1 (macro): {f1:.4f}")
                    metrics_results["f1"] = f1
                except Exception as e2:
                    print(f"Fallback F1 calculation also failed: {e2}")
                    metrics_results["f1"] = float('nan')
        else:
            print("No valid predictions (all 'Unknown'). Skipping classification metrics.")
            metrics_results["accuracy"] = float('nan')
            metrics_results["f1"] = float('nan')

    print("Computing text generation metrics...")
    
    if text_preds and text_refs:
        try:
            # 1) ROUGE
            rouge_result = rouge_metric.compute(predictions=text_preds, references=text_refs)
            print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
            metrics_results["rougeL"] = rouge_result["rougeL"]
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            metrics_results["rougeL"] = float('nan')
        
        try:
            # 2) BLEU
            bleu_result = bleu_metric.compute(predictions=text_preds, references=text_refs)
            print(f"BLEU: {bleu_result['bleu']:.4f}")
            metrics_results["bleu"] = bleu_result["bleu"]
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            metrics_results["bleu"] = float('nan')
        
        try:
            # 3) METEOR
            meteor_result = meteor_metric.compute(predictions=text_preds, references=text_refs)
            print(f"METEOR: {meteor_result['meteor']:.4f}")
            metrics_results["meteor"] = meteor_result["meteor"]
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            metrics_results["meteor"] = float('nan')

 
    # Save to container path
    df_test.to_csv(output_file, index=False)
    print(f"✅ Saved predictions in {output_file}")
    
    # Also save to mounted output directory for local access
    output_path = os.path.join("/outputs", output_file)
    df_test.to_csv(output_path, index=False)
    print(f"✅ Saved predictions locally in {output_path}")
    
    # Save metrics to a separate file for reference
    metrics_file = output_file.replace(".csv", "_metrics.csv")
    metrics_df = pd.DataFrame([metrics_results])
    
    # Save metrics both locally and in container
    metrics_df.to_csv(metrics_file, index=False)
    metrics_df.to_csv(os.path.join("/outputs", metrics_file), index=False)
    print(f"✅ Saved metrics in {metrics_file}")
    
    # Save a token count report for future reference
    token_report_file = output_file.replace(".csv", "_token_report.txt")
    
    # Write to container path
    with open(token_report_file, "w") as f:
        f.write("=== Sequence Length Statistics ===\n")
        lengths = list(dataset_test.sequence_lengths.values())
        f.write(f"Min sequence length: {min(lengths)}\n")
        f.write(f"Max sequence length: {max(lengths)}\n")
        f.write(f"Avg sequence length: {sum(lengths)/len(lengths):.1f}\n")
        f.write(f"Sequences >1000 chars: {sum(1 for l in lengths if l > 1000)}/{len(lengths)}\n")
        f.write(f"Sequences >5000 chars: {sum(1 for l in lengths if l > 5000)}/{len(lengths)}\n\n")
        
        if hasattr(generate_annotation, "token_counts") and generate_annotation.token_counts:
            token_counts = generate_annotation.token_counts
            f.write("=== Token Count Statistics ===\n")
            f.write(f"Min tokens: {min(token_counts)}\n")
            f.write(f"Max tokens: {max(token_counts)}\n")
            f.write(f"Avg tokens: {sum(token_counts)/len(token_counts):.1f}\n")
            f.write(f"Samples with >1000 tokens: {sum(1 for t in token_counts if t > 1000)}/{len(token_counts)}\n")
            f.write(f"Samples with >2000 tokens: {sum(1 for t in token_counts if t > 2000)}/{len(token_counts)}\n")
            f.write(f"Samples with >5000 tokens: {sum(1 for t in token_counts if t > 5000)}/{len(token_counts)}\n\n")
            
            # List the top 10 longest token counts and their indices
            if token_counts:
                f.write("=== Top 10 Longest Token Counts ===\n")
                token_indices = list(range(len(token_counts)))
                longest_indices = sorted(token_indices, key=lambda i: token_counts[i], reverse=True)[:10]
                for i, idx in enumerate(longest_indices):
                    f.write(f"{i+1}. Sample {idx}: {token_counts[idx]} tokens\n")
    
    # Also write to local path
    local_token_report = os.path.join("/outputs", token_report_file)
    with open(local_token_report, "w") as f:
        f.write("=== Sequence Length Statistics ===\n")
        lengths = list(dataset_test.sequence_lengths.values())
        f.write(f"Min sequence length: {min(lengths)}\n")
        f.write(f"Max sequence length: {max(lengths)}\n")
        f.write(f"Avg sequence length: {sum(lengths)/len(lengths):.1f}\n")
        f.write(f"Sequences >1000 chars: {sum(1 for l in lengths if l > 1000)}/{len(lengths)}\n")
        f.write(f"Sequences >5000 chars: {sum(1 for l in lengths if l > 5000)}/{len(lengths)}\n\n")
        
        if hasattr(generate_annotation, "token_counts") and generate_annotation.token_counts:
            token_counts = generate_annotation.token_counts
            f.write("=== Token Count Statistics ===\n")
            f.write(f"Min tokens: {min(token_counts)}\n")
            f.write(f"Max tokens: {max(token_counts)}\n")
            f.write(f"Avg tokens: {sum(token_counts)/len(token_counts):.1f}\n")
            f.write(f"Samples with >1000 tokens: {sum(1 for t in token_counts if t > 1000)}/{len(token_counts)}\n")
            f.write(f"Samples with >2000 tokens: {sum(1 for t in token_counts if t > 2000)}/{len(token_counts)}\n")
            f.write(f"Samples with >5000 tokens: {sum(1 for t in token_counts if t > 5000)}/{len(token_counts)}\n\n")
            
            # List the top 10 longest token counts and their indices
            if token_counts:
                f.write("=== Top 10 Longest Token Counts ===\n")
                token_indices = list(range(len(token_counts)))
                longest_indices = sorted(token_indices, key=lambda i: token_counts[i], reverse=True)[:10]
                for i, idx in enumerate(longest_indices):
                    f.write(f"{i+1}. Sample {idx}: {token_counts[idx]} tokens\n")
    
    print(f"✅ Saved token report in {token_report_file}")
    
    print("\n=== Final Metrics Summary ===")
    for metric, value in metrics_results.items():
        print(f"{metric}: {value:.4f}")

    return "Baseline inference completed successfully."


if __name__ == "__main__":
    pass