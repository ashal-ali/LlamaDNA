import modal
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

# Create Modal App
app = modal.App("dna-llama-finetune")
volume = modal.NetworkFileSystem.from_name("dna-llama-data", create_if_missing=True)

class GenomicDataset(Dataset):
    def __init__(self, df, dna_tokenizer, llama_tokenizer, mode="train"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.dna_tokenizer = dna_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.mode = mode
        
        print(f"Loaded {mode} dataset with {len(df)} samples")
        
        required_columns = ["Sequence", "dataset", "Gene name", "Phenotype description", "clinSign"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in dataset!")
        
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
        row = self.df.iloc[idx]
        
        dna_seq = row["Sequence"]
        if len(dna_seq) > 5000:
            dna_seq = dna_seq[:5000]
        
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
        
        ground_annotation = row.get("Ground_Annotation", "")
        dataset_label = row.get("dataset", "")
        
        target_text = f"This sequence is found in the {dataset_label} dataset. {ground_annotation}"
        
        return {
            "dna_sequence": dna_seq,
            "prompt": prompt,
            "target_text": target_text
        }

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_epoch(dnabert_model, projection_layer, llama_model, train_loader, 
                dna_tokenizer, llama_tokenizer, optimizer, device):
    projection_layer.train()
    llama_model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        dna_sequences = batch["dna_sequence"]
        prompts = batch["prompt"]
        target_texts = batch["target_text"]
        
        with torch.no_grad():
            dna_inputs = dna_tokenizer(dna_sequences, return_tensors="pt", 
                                       padding=True, truncation=True).to(device)
            dna_outputs = dnabert_model(**dna_inputs)
            dna_embeddings = dna_outputs[0]
        
        projected_embeddings = projection_layer(dna_embeddings)
        
        text_inputs = llama_tokenizer(prompts, return_tensors="pt", 
                                     padding=True, truncation=True).to(device)
        target_inputs = llama_tokenizer(target_texts, return_tensors="pt", 
                                       padding=True, truncation=True).to(device)
        
        text_embeddings = llama_model.get_input_embeddings()(text_inputs.input_ids)
        
        combined_embeddings = torch.cat([
            projected_embeddings.half(), 
            text_embeddings.half()
        ], dim=1)
        
        dna_seq_len = projected_embeddings.size(1)
        text_seq_len = text_embeddings.size(1)
        batch_size = combined_embeddings.size(0)
        
        labels = torch.full(
            (batch_size, dna_seq_len + text_seq_len),
            -100,
            device=device
        )
        
        labels[:, dna_seq_len:] = target_inputs.input_ids
        
        outputs = llama_model(
            inputs_embeds=combined_embeddings,
            labels=labels
        )
        
        loss = outputs.loss
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return epoch_loss / len(train_loader)

def evaluate_model(dnabert_model, projection_layer, llama_model, val_loader,
                  dna_tokenizer, llama_tokenizer, device):
    projection_layer.eval()
    llama_model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            dna_sequences = batch["dna_sequence"]
            prompts = batch["prompt"]
            target_texts = batch["target_text"]
            
            dna_inputs = dna_tokenizer(dna_sequences, return_tensors="pt", 
                                      padding=True, truncation=True).to(device)
            dna_outputs = dnabert_model(**dna_inputs)
            dna_embeddings = dna_outputs[0]
            
            projected_embeddings = projection_layer(dna_embeddings)
            
            text_inputs = llama_tokenizer(prompts, return_tensors="pt", 
                                         padding=True, truncation=True).to(device)
            target_inputs = llama_tokenizer(target_texts, return_tensors="pt", 
                                           padding=True, truncation=True).to(device)
            
            text_embeddings = llama_model.get_input_embeddings()(text_inputs.input_ids)
            
            combined_embeddings = torch.cat([
                projected_embeddings.half(), 
                text_embeddings.half()
            ], dim=1)
            
            dna_seq_len = projected_embeddings.size(1)
            text_seq_len = text_embeddings.size(1)
            batch_size = combined_embeddings.size(0)
            
            labels = torch.full(
                (batch_size, dna_seq_len + text_seq_len),
                -100,
                device=device
            )
            labels[:, dna_seq_len:] = target_inputs.input_ids
            
            outputs = llama_model(
                inputs_embeds=combined_embeddings,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Validation Batch {batch_idx + 1}/{len(val_loader)}, Loss: {outputs.loss.item():.4f}")
    
    return val_loss / len(val_loader)

@app.function(
    gpu="A100",
    timeout=3600,
    network_file_systems={"/data": volume},
    image=modal.Image.debian_slim().apt_install("pkg-config", "cmake").pip_install(
        "torch",
        "transformers>=4.35.0",
        "einops",
        "sentencepiece",
        "accelerate>=0.26.0",
        "pandas",
        "scikit-learn",
        "peft>=0.5.0",
    )
)
def finetune_with_lora():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        os.makedirs("/data/model", exist_ok=True)
        
        print("Loading training data...")
        train_file = "/data/dataset_train.csv"
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        df_train = pd.read_csv(train_file)
        print(f"Loaded training data with {len(df_train)} rows and columns: {df_train.columns.tolist()}")
        
        train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)
        print(f"Split into {len(train_df)} training and {len(val_df)} validation samples")
        
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
        
        print("Configuring LoRA for Llama model...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            inference_mode=False
        )
        
        llama_model = get_peft_model(llama_model, lora_config)
        llama_model.print_trainable_parameters()
        
        projection_layer = ProjectionLayer(
            input_dim=dnabert_model.config.hidden_size,
            output_dim=llama_model.config.hidden_size
        ).to(device)
        
        train_dataset = GenomicDataset(train_df, dna_tokenizer, llama_tokenizer, mode="train")
        val_dataset = GenomicDataset(val_df, dna_tokenizer, llama_tokenizer, mode="val")
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        optimizer = optim.AdamW([
            {"params": projection_layer.parameters()},
            {"params": llama_model.parameters()}
        ], lr=1e-4)
        
        best_loss = float('inf')
        
        for epoch in range(3):
            print(f"\n==== Epoch {epoch+1}/3 ====")
            
            train_loss = train_epoch(
                dnabert_model, projection_layer, llama_model, train_loader,
                dna_tokenizer, llama_tokenizer, optimizer, device
            )
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss = evaluate_model(
                dnabert_model, projection_layer, llama_model, val_loader,
                dna_tokenizer, llama_tokenizer, device
            )
            print(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"New best model (loss: {best_loss:.4f})! Saving...")
                
                torch.save(projection_layer.state_dict(), "/data/model/projection_layer.pt")
                
                llama_model.save_pretrained("/data/model/llama_lora_adapter")
                
                with open("/data/model/checkpoint_info.txt", "w") as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write(f"Train Loss: {train_loss:.6f}\n")
                    f.write(f"Validation Loss: {val_loss:.6f}\n")
                    f.write(f"DNABERT model: {dnabert_name}\n")
                    f.write(f"Llama model: {llama_name}\n")
        
        print(f"\nTraining complete! Best validation loss: {best_loss:.4f}")
        print(f"Model saved to: /data/model/")
        
        print("\nGenerating a sample prediction...")
        sample_dna = val_df.iloc[0]["Sequence"][:500]
        sample_prompt = "Generate a functional annotation for this DNA sequence."
        
        with torch.no_grad():
            dna_inputs = dna_tokenizer(sample_dna, return_tensors="pt").to(device)
            dna_outputs = dnabert_model(**dna_inputs)
            dna_embeddings = dna_outputs[0]
            projected_embeddings = projection_layer(dna_embeddings)
            
            text_inputs = llama_tokenizer(sample_prompt, return_tensors="pt").to(device)
            text_embeddings = llama_model.get_input_embeddings()(text_inputs.input_ids)
            
            combined_embeddings = torch.cat([
                projected_embeddings.half(),
                text_embeddings.half()
            ], dim=1)
            
            generated_ids = llama_model.generate(
                inputs_embeds=combined_embeddings,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_text = llama_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Sample prediction: {generated_text}")
        
        return {
            "status": "success",
            "best_loss": float(best_loss),
            "model_path": "/data/model/"
        }
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }

@app.local_entrypoint()
def main():
    print("Starting fine-tuning setup...")
    
    with volume.get_client() as vol_client:
        if not vol_client.exists("/data"):
            vol_client.mkdir("/data")
        if not vol_client.exists("/data/model"):
            vol_client.mkdir("/data/model")
    
    with volume.get_client() as vol_client:
        print("Uploading dataset_train.csv to Modal volume...")
        try:
            if not os.path.exists("dataset_train.csv"):
                print("Error: dataset_train.csv not found in current directory!")
                return
            
            with open("dataset_train.csv", "rb") as f:
                vol_client.write("/data/dataset_train.csv", f.read())
            
            print("Successfully uploaded dataset_train.csv to volume")
        except Exception as e:
            print(f"Error uploading file: {e}")
            return
    
    print("Starting fine-tuning on Modal...")
    result = finetune_with_lora.remote()
    print(f"Fine-tuning complete: {result}")
    
    print("Downloading trained models from Modal volume...")
    with volume.get_client() as vol_client:
        if vol_client.exists("/data/model/projection_layer.pt"):
            model_data = vol_client.read("/data/model/projection_layer.pt")
            with open("projection_layer.pt", "wb") as f:
                f.write(model_data)
            print("Successfully downloaded projection layer to projection_layer.pt")
        
        if vol_client.exists("/data/model/llama_lora_adapter"):
            os.makedirs("llama_lora_adapter", exist_ok=True)
            
            lora_files = vol_client.listdir("/data/model/llama_lora_adapter")
            
            for file_name in lora_files:
                file_path = f"/data/model/llama_lora_adapter/{file_name}"
                file_data = vol_client.read(file_path)
                
                with open(f"llama_lora_adapter/{file_name}", "wb") as f:
                    f.write(file_data)
            
            print("Successfully downloaded LoRA adapter to llama_lora_adapter/")
        
        if vol_client.exists("/data/model/checkpoint_info.txt"):
            info_data = vol_client.read("/data/model/checkpoint_info.txt")
            with open("checkpoint_info.txt", "wb") as f:
                f.write(info_data)
            print("Successfully downloaded checkpoint info to checkpoint_info.txt")
        else:
            print("Warning: Model files not found in volume. Training might have failed.")