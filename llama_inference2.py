import modal
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import evaluate
import os
from modal import Mount

app = modal.App("llama-dna-inference")

test_file_path = "dataset_test.csv"
remote_test_file_path = "/root/dataset_test.csv"
output_dir = "."
local_mount = Mount.from_local_dir(output_dir, remote_path="/outputs")

image = (
    modal.Image.debian_slim()
    .apt_install("pkg-config", "cmake")
    .pip_install(
        "transformers>=4.35.0",
        "accelerate>=0.26.0",
        "torch",
        "pandas",
        "evaluate",
        "scikit-learn",
        "sentencepiece",
        "rouge_score",
        "nltk",
        "sacrebleu",
        "python-dateutil",
        "peft>=0.5.0",
    )
    .add_local_file(test_file_path, remote_path=remote_test_file_path)
)

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class LlamaDNATestDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        
        self.sequence_lengths = {idx: len(row.get("Sequence", "")) for idx, row in df.iterrows()}
        lengths = list(self.sequence_lengths.values())
        print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        
        self.prompt_template = (
            "DNA Sequence Analysis:\n"
            "Given the following information:\n"
            "- Gene name: {gene_name}\n"
            "- Gene description: {gene_desc}\n"
            "- Phenotype: {phenotype}\n"
            "- Clinical significance: {clinical_sig}\n"
            "- SPLASH scores: {splash_scores}\n\n"
            "Generate a functional annotation for the DNA sequence."
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        sequence = row.get("Sequence", "")
        
        MAX_SEQUENCE_LENGTH = 5000
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
        
        gene_name = row.get("Gene name", "Unknown")
        gene_desc = row.get("Gene description", "Unknown")
        phenotype = row.get("Phenotype description", "Unknown")
        clinical_sig = row.get("clinSign", "Unknown")
        
        splash_effect = row.get("SPLASH_Effect", 0.5)
        splash_pval = row.get("SPLASH_pval", 0.5)
        splash_entropy = row.get("SPLASH_entropy", 0.5)
        splash_scores = f"{splash_effect:.4f}, {splash_pval:.4f}, {splash_entropy:.4f}"
        
        prompt = self.prompt_template.format(
            gene_name=gene_name,
            gene_desc=gene_desc,
            phenotype=phenotype,
            clinical_sig=clinical_sig,
            splash_scores=splash_scores
        )
        
        return {
            "sequence": sequence,
            "prompt": prompt,
            "idx": idx
        }

@app.function(image=image, gpu="A100", timeout=600, mounts=[local_mount])
def run_llama_dna_inference(
    test_file="/root/dataset_test.csv",
    output_file="llama_dna_predictions.csv",
    projection_model_path="/model/projection_layer.pt",
    lora_model_path="/model/llama_lora_adapter",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading test dataset from {test_file}")
    
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found at {test_file}")
        return "Error: Test file not found."
    
    df_test = pd.read_csv(test_file)
    print(f"Loaded test dataset with {len(df_test)} rows")
    
    # Load DNABERT model
    print("Loading DNABERT-2...")
    dnabert_name = "zhihan1996/DNABERT-2-117M"
    dna_tokenizer = AutoTokenizer.from_pretrained(dnabert_name)
    dnabert_model = AutoModel.from_pretrained(dnabert_name, trust_remote_code=True).to(device)
    dnabert_model.eval()
    
    print("Loading Llama-2-7b...")
    llama_name = "NousResearch/Llama-2-7b-chat-hf"
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_name)
    llama_model = LlamaForCausalLM.from_pretrained(
        llama_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from {lora_model_path}")
    if os.path.exists(lora_model_path):
        llama_model = PeftModel.from_pretrained(llama_model, lora_model_path)
        print("Successfully loaded LoRA adapter")
    else:
        print(f"WARNING: LoRA adapter not found at {lora_model_path}, using base model")
    print(f"Loading projection layer from {projection_model_path}")
    projection_layer = ProjectionLayer(
        input_dim=dnabert_model.config.hidden_size,
        output_dim=llama_model.config.hidden_size
    ).to(device)
    
    if os.path.exists(projection_model_path):
        projection_layer.load_state_dict(torch.load(projection_model_path, map_location=device))
        print("Successfully loaded projection layer")
    else:
        print(f"WARNING: Projection layer not found at {projection_model_path}, using random initialization")
    
    projection_layer.eval()
    llama_model.eval()
    dataset = LlamaDNATestDataset(df_test)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
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
    
    def extract_dataset_label(annotation_text):
        txt_lower = annotation_text.lower()
        
        if "clinvar" in txt_lower:
            return "ClinVar"
        elif "cosmic" in txt_lower:
            return "COSMIC"
            
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
    
    for idx, batch in enumerate(loader):
        sequence = batch["sequence"][0]
        prompt = batch["prompt"][0]
        row_idx = batch["idx"].item()
        
        if idx % 10 == 0:
            print(f"Processing sample {idx}/{len(loader)}...")
        
        try:
            with torch.no_grad():
                dna_inputs = dna_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True).to(device)
                dna_outputs = dnabert_model(**dna_inputs)
                dna_embeddings = dna_outputs[0]
            with torch.no_grad():
                projected_embeddings = projection_layer(dna_embeddings)
            text_inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
            text_embeddings = llama_model.get_input_embeddings()(text_inputs.input_ids)
            combined_embeddings = torch.cat([
                projected_embeddings.half(),
                text_embeddings.half()
            ], dim=1)
            with torch.no_grad():
                generated_ids = llama_model.generate(
                    inputs_embeds=combined_embeddings,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            generated_text = llama_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            prompt_text = llama_tokenizer.decode(text_inputs.input_ids[0], skip_special_tokens=True)
            if prompt_text in generated_text:
                annotation_pred = generated_text.replace(prompt_text, "").strip()
            else:
                annotation_pred = generated_text.strip()
            
            dataset_pred = extract_dataset_label(annotation_pred)
            df_test.at[row_idx, "Predicted_Annotation"] = annotation_pred
            df_test.at[row_idx, "Predicted_Dataset"] = dataset_pred
            
            if "dataset" in df_test.columns:
                class_preds.append(dataset_pred)
                class_refs.append(df_test.at[row_idx, "dataset"])
            
            if "Ground_Annotation" in df_test.columns:
                text_preds.append(annotation_pred)
                text_refs.append(df_test.at[row_idx, "Ground_Annotation"])
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            df_test.at[row_idx, "Predicted_Annotation"] = f"ERROR: {str(e)}"
            df_test.at[row_idx, "Predicted_Dataset"] = "Unknown"
    
    print("Computing metrics...")
    metrics_results = {}
    
    if class_preds and class_refs:
        valid_indices = [i for i, pred in enumerate(class_preds) if pred != "Unknown"]
        filtered_preds = [class_preds[i] for i in valid_indices]
        filtered_refs = [class_refs[i] for i in valid_indices]
        
        if filtered_preds:
            # Compute accuracy
            correct = sum(1 for p, r in zip(filtered_preds, filtered_refs) if p == r)
            accuracy = correct / len(filtered_preds)
            print(f"Classification Accuracy: {accuracy:.4f}")
            metrics_results["accuracy"] = accuracy
            
            # Compute F1 score
            from sklearn.metrics import f1_score
            try:
                unique_labels = sorted(set(filtered_refs + filtered_preds))
                label_to_idx = {label: i for i, label in enumerate(unique_labels)}
                idx_refs = [label_to_idx[label] for label in filtered_refs]
                idx_preds = [label_to_idx[label] for label in filtered_preds]
                
                f1 = f1_score(idx_refs, idx_preds, average="macro")
                print(f"Classification F1 (macro): {f1:.4f}")
                metrics_results["f1"] = f1
            except Exception as e:
                print(f"Error calculating F1 score: {e}")
                metrics_results["f1"] = float('nan')
        else:
            print("No valid predictions (all 'Unknown'). Skipping classification metrics.")
            metrics_results["accuracy"] = float('nan')
            metrics_results["f1"] = float('nan')
    
    # Compute text generation metrics
    if text_preds and text_refs:
        try:
            # ROUGE
            rouge_result = rouge_metric.compute(predictions=text_preds, references=text_refs)
            print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")
            metrics_results["rougeL"] = rouge_result["rougeL"]
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            metrics_results["rougeL"] = float('nan')
        
        try:
            # BLEU
            bleu_result = bleu_metric.compute(predictions=text_preds, references=[[ref] for ref in text_refs])
            print(f"BLEU: {bleu_result['bleu']:.4f}")
            metrics_results["bleu"] = bleu_result["bleu"]
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            metrics_results["bleu"] = float('nan')
        
        try:
            # METEOR
            meteor_result = meteor_metric.compute(predictions=text_preds, references=text_refs)
            print(f"METEOR: {meteor_result['meteor']:.4f}")
            metrics_results["meteor"] = meteor_result["meteor"]
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            metrics_results["meteor"] = float('nan')
    df_test.to_csv(output_file, index=False)
    print(f"✅ Saved predictions in {output_file}")
    
    output_path = os.path.join("/outputs", output_file)
    df_test.to_csv(output_path, index=False)
    print(f"✅ Saved predictions locally in {output_path}")
    metrics_file = output_file.replace(".csv", "_metrics.csv")
    metrics_df = pd.DataFrame([metrics_results])
    
    metrics_df.to_csv(metrics_file, index=False)
    metrics_df.to_csv(os.path.join("/outputs", metrics_file), index=False)
    print(f"✅ Saved metrics in {metrics_file}")
    
    print("\n=== Final Metrics Summary ===")
    for metric, value in metrics_results.items():
        print(f"{metric}: {value:.4f}")
    
    return "Llama-DNA inference completed successfully."

@app.local_entrypoint()
def main():
    print("Starting Llama-DNA inference...")
    local_projection_path = "projection_layer.pt"
    local_lora_path = "llama_lora_adapter"
    
    if not os.path.exists(local_projection_path):
        print(f"Warning: Projection layer not found at {local_projection_path}")
    
    if not os.path.exists(local_lora_path):
        print(f"Warning: LoRA adapter not found at {local_lora_path}")
    
    with app.volume.get_client() as vol_client:
        vol_client.mkdir("/model", exist_ok=True)
    
    with app.volume.get_client() as vol_client:
        if os.path.exists(local_projection_path):
            with open(local_projection_path, "rb") as f:
                vol_client.write("/model/projection_layer.pt", f.read())
            print("Successfully uploaded projection layer")
        
        if os.path.exists(local_lora_path) and os.path.isdir(local_lora_path):
            vol_client.mkdir("/model/llama_lora_adapter", exist_ok=True)
            
            for file_name in os.listdir(local_lora_path):
                file_path = os.path.join(local_lora_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, "rb") as f:
                        vol_client.write(f"/model/llama_lora_adapter/{file_name}", f.read())
            
            print("Successfully uploaded LoRA adapter")
    
    print("Starting inference on Modal...")
    result = run_llama_dna_inference.remote()
    print(f"Inference complete: {result}")

if __name__ == "__main__":
    main()