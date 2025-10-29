import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

DATA_PATH = "../dataset_train.csv"

MODEL_NAME = "google/gemma-2-2b"

RUN_ROOT = Path("runs_ft_gemma")
RUN_DIR = RUN_ROOT / MODEL_NAME.replace("/", "__")
MAX_LEN = 512

USE_LORA = True
USE_8BIT = True
EPOCHS = 1
LR_LORA = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

PER_DEVICE_TRAIN_BS = 1
PER_DEVICE_EVAL_BS  = 4
GRAD_ACCUM = 16
SEED = 42

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def cuda_diag():
    print_section("CUDA DIAGNOSTIC")
    print(f"torch: {torch.__version__}, torch.version.cuda: {torch.version.cuda}")
    print(f"cuda.is_available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU[0]: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM total: {mem_gb:.1f} GB")
        except Exception as e:
            print("get_device_name error:", e)

def load_data(csv_path: str):
    print_section("LOADING DATA")
    df = pd.read_csv(csv_path)
    
    texts = []
    for _, row in df.iterrows():
        name = str(row["movie_name"]).strip()
        desc = str(row["description"]).strip()
        text = f"Classify the genres of this movie:\nTitle: {name}\nDescription: {desc}"
        texts.append(text)
    
    labels_raw = df["genre"].apply(lambda s: [g.strip() for g in str(s).split(",") if g.strip()])
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_raw)
    
    X_tr, X_va, Y_tr, Y_va = train_test_split(texts, Y, test_size=0.2, random_state=SEED)
    print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Labels: {len(mlb.classes_)}")
    print(f"Géneros: {', '.join(mlb.classes_[:10])}{'...' if len(mlb.classes_) > 10 else ''}")
    return X_tr, X_va, Y_tr.astype(np.float32), Y_va.astype(np.float32), mlb

def prepare_tokenizer():
    print_section("TOKENIZER")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    
    print(f"Tokenizer loaded. vocab={tok.vocab_size} | pad_token_id={tok.pad_token_id}")
    return tok

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
    
    def __len__(self): 
        return len(self.texts)
    
    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], 
            truncation=True, 
            max_length=self.max_len,
            padding=False
        )
        enc["labels"] = self.labels[i]
        return enc

def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(axis=0)
    N = y.shape[0]
    pos = np.clip(pos, 1.0, None)
    w = (N - pos) / pos
    return torch.tensor(w, dtype=torch.float32)

class BCEWithPosWeightTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def build_model(num_labels: int, tokenizer):
    print_section("MODEL")
    
    quant = None
    if USE_8BIT:
        try:
            quant = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            print("Cuantización int8 activada")
        except Exception as e:
            print(f"bitsandbytes error: {e}")
            quant = None

    print(f"Cargando {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        quantization_config=quant,
        device_map="auto" if quant else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not quant else None,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing activado")

    if USE_LORA:
        if quant:
            model = prepare_model_for_kbit_training(model)
        
        lconf = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params totales: {total_params/1e6:.1f}M")
    print(f"Params entrenables: {trainable/1e6:.1f}M ({trainable/total_params*100:.2f}%)")
    print(f"num_labels: {num_labels} | LoRA: {USE_LORA}")
    
    return model

def compute_metrics_default(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

def optimize_thresholds(logits: np.ndarray, y_true: np.ndarray):
    print_section("OPTIMIZING PER-CLASS THRESHOLDS")
    probs = 1.0 / (1.0 + np.exp(-logits))
    K = probs.shape[1]
    thresholds = np.zeros(K, dtype=np.float32)
    qs = np.linspace(0.01, 0.99, 50)
    
    for k in range(K):
        s = probs[:, k]
        best_f1, best_t = 0.0, 0.5
        grid = np.quantile(s, qs, method="linear")
        for t in grid:
            pred = (s >= t).astype(int)
            f1 = f1_score(y_true[:, k], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
    
    print("Thresholds optimized.")
    return thresholds

def evaluate_with_thresholds(logits: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    micro = f1_score(y_true, preds, average="micro", zero_division=0)
    macro = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted = f1_score(y_true, preds, average="weighted", zero_division=0)
    return {"micro_f1": float(micro), "macro_f1": float(macro), "weighted_f1": float(weighted)}

def main():
    set_seed(SEED)
    print_section("GEMMA-2B LoRA MULTILABEL TRAIN")
    cuda_diag()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = RUN_DIR / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    X_tr, X_va, Y_tr, Y_va, mlb = load_data(DATA_PATH)
    pos_weight = compute_pos_weight(Y_tr)

    tokenizer = prepare_tokenizer()
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    ds_tr = MovieDataset(X_tr, Y_tr, tokenizer, MAX_LEN)
    ds_va = MovieDataset(X_va, Y_va, tokenizer, MAX_LEN)

    model = build_model(num_labels=Y_tr.shape[1], tokenizer=tokenizer)

    print_section("TRAINING")
    args = TrainingArguments(
        output_dir=str(RUN_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LR_LORA,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
        remove_unused_columns=True,
        optim="adamw_torch",
    )

    trainer = BCEWithPosWeightTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_default,
        pos_weight=pos_weight,
    )

    print("Iniciando entrenamiento...")
    trainer.train()

    print_section("EVALUATION (threshold=0.5)")
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.6f}")

    print("Generando predicciones para optimización...")
    preds = trainer.predict(ds_va)
    logits_va = preds.predictions
    thresholds = optimize_thresholds(logits_va, Y_va)
    tuned = evaluate_with_thresholds(logits_va, Y_va, thresholds)

    print_section("EVALUATION (per-class optimized thresholds)")
    for k, v in tuned.items():
        print(f"{k}: {v:.6f}")

    print_section("SAVING ARTIFACTS")
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    
    with open(RUN_DIR / "labels.json", "w") as f:
        json.dump(mlb.classes_.tolist(), f, indent=2)
    np.save(RUN_DIR / "thresholds.npy", thresholds)
    
    with open(RUN_DIR / "metrics_default.json", "w") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (float, int))}, f, indent=2)
    with open(RUN_DIR / "metrics_tuned.json", "w") as f:
        json.dump(tuned, f, indent=2)
    
    config = {
        "model": MODEL_NAME,
        "use_lora": USE_LORA,
        "use_8bit": USE_8BIT,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "epochs": EPOCHS,
        "lr": LR_LORA,
        "max_len": MAX_LEN,
    }
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Artifacts in: {RUN_DIR.resolve()}")
    print(f"- Model/LoRA:     {best_dir}")
    print(f"- Tokenizer:      {best_dir}")
    print(f"- Labels:         {RUN_DIR/'labels.json'}")
    print(f"- Thresholds:     {RUN_DIR/'thresholds.npy'}")
    print(f"- Metrics def.:   {RUN_DIR/'metrics_default.json'}")
    print(f"- Metrics tuned:  {RUN_DIR/'metrics_tuned.json'}")
    print(f"- Config:         {RUN_DIR/'config.json'}")

if __name__ == "__main__":
    main()