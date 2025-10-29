"""
Fine-tuning multilabel (movie genres) with MORE POWERFUL models + LoRA (8-bit)
Models to try (in order of power):
1. microsoft/deberta-v3-large (435M params) - often SOTA for text classification
2. roberta-large (355M params) - strong baseline
3. google/electra-large-discriminator (335M params) - efficient and powerful
4. microsoft/deberta-v3-base (184M params) - good balance if VRAM limited

Changes for better macro_f1:
- Focal Loss option (helps with hard classes)
- Class-balanced loss weighting
- Longer training with patience
- Lower learning rate for stability
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ===================== Config =====================
DATA_PATH = "../dataset_train.csv"

# 🔥 CHOOSE YOUR MODEL (uncomment one):
MODEL_NAME = "microsoft/deberta-v3-large"  # 🥇 Best for macro_f1, needs ~14GB VRAM
# MODEL_NAME = "roberta-large"              # 🥈 Strong alternative, ~12GB VRAM
# MODEL_NAME = "google/electra-large-discriminator"  # 🥉 Fast + powerful
# MODEL_NAME = "microsoft/deberta-v3-base"  # 💚 If VRAM limited (~10GB)

RUN_ROOT = Path("runs_ft_powerful")
RUN_DIR = RUN_ROOT / MODEL_NAME.replace("/", "__")
MAX_LEN = 512

# Hiperparámetros optimizados para macro_f1
USE_LORA = True
USE_8BIT = False
EPOCHS = 10              # Más épocas con early stopping
LR_LORA = 1e-4           # LR más baja para estabilidad con modelos grandes
LR_FULL = 5e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1       # Más warmup para modelos grandes

# Loss function: "bce" (default) o "focal" (mejor para clases difíciles)
LOSS_TYPE = "focal"      # 🔥 Focal loss ayuda con macro_f1
FOCAL_ALPHA = 0.25       # Balance entre clases
FOCAL_GAMMA = 2.0        # Focus en ejemplos difíciles

# VRAM budget
PER_DEVICE_TRAIN_BS = 2
PER_DEVICE_EVAL_BS = 8
GRAD_ACCUM = 8
SEED = 42

# LoRA config para modelos grandes
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1      # Más dropout para regularización

# Target modules por modelo
TARGET_MODULES = {
    "deberta": ["query_proj", "key_proj", "value_proj", "dense"],
    "roberta": ["query", "key", "value", "dense"],
    "electra": ["query", "key", "value", "dense"],
}

EARLY_STOPPING_PATIENCE = 3

# ==================================================

def get_target_modules(model_name: str):
    """Detecta los módulos correctos según el modelo"""
    if "deberta" in model_name.lower():
        return TARGET_MODULES["deberta"]
    elif "roberta" in model_name.lower():
        return TARGET_MODULES["roberta"]
    elif "electra" in model_name.lower():
        return TARGET_MODULES["electra"]
    else:
        return TARGET_MODULES["roberta"]  # default

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def cuda_diag():
    print_section("🧪 CUDA DIAGNOSTIC")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
    print(f"cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def load_data(csv_path: str):
    print_section("📂 LOADING DATA")
    df = pd.read_csv(csv_path)
    texts = (df["movie_name"].fillna("") + " [SEP] " + df["description"].fillna("")).tolist()
    labels_raw = df["genre"].apply(lambda s: [g.strip() for g in str(s).split(",") if g.strip()])
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_raw)
    
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        texts, Y, test_size=0.2, random_state=SEED
    )
    
    print(f"✓ Train: {len(X_tr)} | Val: {len(X_va)} | Labels: {len(mlb.classes_)}")
    
    print("\nClass distribution (train):")
    class_counts = Y_tr.sum(axis=0)
    for i, cls in enumerate(mlb.classes_):
        print(f"  {cls}: {int(class_counts[i])} ({100*class_counts[i]/len(Y_tr):.1f}%)")
    
    return X_tr, X_va, Y_tr.astype(np.float32), Y_va.astype(np.float32), mlb

def prepare_tokenizer():
    print_section("🔤 TOKENIZER")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    
    print(f"✓ Tokenizer: {tok.__class__.__name__}")
    print(f"  vocab_size={tok.vocab_size} | pad_token_id={tok.pad_token_id}")
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
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        enc["labels"] = self.labels[i]
        return enc

def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """pos_weight para BCE: (N - pos) / pos"""
    pos = y.sum(axis=0)
    N = y.shape[0]
    pos = np.clip(pos, 1.0, None)
    w = (N - pos) / pos
    return torch.tensor(w, dtype=torch.float32)

class FocalLoss(torch.nn.Module):
    """Focal Loss para multilabel - ayuda con clases difíciles (mejor macro_f1)"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

class CustomLossTrainer(Trainer):
    def __init__(self, loss_type="bce", pos_weight=None, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        if loss_type == "focal":
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.loss_type == "focal":
            loss = self.focal_loss(logits, labels)
        else:  # bce
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            )
            loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def build_model(num_labels: int, tokenizer):
    print_section("🤖 MODEL")
    quant = None
    if USE_8BIT:
        try:
            quant = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            quant = None
            print("⚠️ bitsandbytes no disponible: se cargará en 16/32 bits.")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        quantization_config=quant,
        device_map={"":0} if quant is not None else None,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if USE_LORA:
        model = prepare_model_for_kbit_training(model) if quant is not None else model
        target_mods = get_target_modules(MODEL_NAME)
        
        lconf = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=target_mods,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters()
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
    return model

def compute_metrics_default(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    
    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    
    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "min_class_f1": float(per_class.min()),
        "max_class_f1": float(per_class.max()),
    }

def optimize_thresholds(logits: np.ndarray, y_true: np.ndarray):
    print_section("🎚️  OPTIMIZING PER-CLASS THRESHOLDS")
    probs = 1.0 / (1.0 + np.exp(-logits))
    K = probs.shape[1]
    thresholds = np.zeros(K, dtype=np.float32)
    
    for k in range(K):
        s = probs[:, k]
        best_f1, best_t = 0.0, 0.5
        grid = np.linspace(0.01, 0.99, 199)
        for t in grid:
            pred = (s >= t).astype(int)
            f1 = f1_score(y_true[:, k], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
        if best_f1 < 0.3:
            print(f"  ⚠️ Class {k}: F1={best_f1:.3f}, threshold={best_t:.3f}")
    print("✓ Thresholds optimized.")
    return thresholds

def evaluate_with_thresholds(logits: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    micro = f1_score(y_true, preds, average="micro", zero_division=0)
    macro = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted = f1_score(y_true, preds, average="weighted", zero_division=0)
    return {"micro_f1": float(micro), "macro_f1": float(macro), "weighted_f1": float(weighted)}

# ===================== PREDICCIÓN 100% EN MEMORIA =====================
_GLOBAL_PREDICTOR = {"model": None, "tokenizer": None, "thresholds": None, "labels": None}

def prepare_predictor(model, tokenizer, thresholds: np.ndarray, labels: list[str]):
    """
    Registra en memoria el predictor para usarlo sin cargar/guardar.
    """
    _GLOBAL_PREDICTOR["model"] = model
    _GLOBAL_PREDICTOR["tokenizer"] = tokenizer
    _GLOBAL_PREDICTOR["thresholds"] = thresholds.astype(np.float32)
    _GLOBAL_PREDICTOR["labels"] = list(labels)

def _predict_texts(texts, model, tokenizer, thresholds, batch_size=64, max_len=512):
    """Devuelve (pred_bin, probs) para una lista de textos usando el modelo en memoria."""
    model.eval()
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, texts): self.texts = texts
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return tokenizer(self.texts[i], truncation=True, max_length=max_len)

    dl = torch.utils.data.DataLoader(_DS(texts), batch_size=batch_size, shuffle=False, collate_fn=collator)
    all_logits = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred_bin = (probs >= thresholds.reshape(1, -1)).astype(int)
    return pred_bin, probs

def predict_in_memory(input_source, return_probs: bool = False, batch_size: int = 64, max_len: int = MAX_LEN):
    """
    Predice SIN cargar/guardar nada.
    - input_source: ruta CSV o DataFrame con columnas ['movie_name','description'].
    - Usa el modelo/tokenizer/thresholds/labels ya registrados con prepare_predictor(...).
    - Devuelve un DataFrame con columnas ['movie_name','genre','description'] (+ probs opcional).
    """
    assert _GLOBAL_PREDICTOR["model"] is not None, "Llama a prepare_predictor(model, tokenizer, thresholds, labels) antes."
    model = _GLOBAL_PREDICTOR["model"]
    tokenizer = _GLOBAL_PREDICTOR["tokenizer"]
    thresholds = _GLOBAL_PREDICTOR["thresholds"]
    labels = _GLOBAL_PREDICTOR["labels"]

    if isinstance(input_source, (str, Path)):
        df = pd.read_csv(input_source)
    else:
        df = input_source.copy()

    texts = (df["movie_name"].fillna("") + " [SEP] " + df["description"].fillna("")).tolist()
    pred_bin, probs = _predict_texts(texts, model, tokenizer, thresholds, batch_size=batch_size, max_len=max_len)

    pred_labels = [", ".join([labels[j] for j, v in enumerate(row) if v == 1]) for row in pred_bin]
    out_df = pd.DataFrame({
        "movie_name": df.get("movie_name", pd.Series([""] * len(texts))),
        "genre": pred_labels,
        "description": df.get("description", pd.Series([""] * len(texts))),
    })
    # save df
    out_df.to_csv(f"../dataset_test_predictions_ENTREGAAAA.csv", index=False)
    if return_probs:
        for j, lab in enumerate(labels):
            out_df[f"prob_{lab}"] = probs[:, j]
    return out_df, pred_bin, probs

# ===================== MAIN =====================
def main():
    set_seed(SEED)
    print_section(f"🚀 POWERFUL MODEL TRAINING: {MODEL_NAME}")
    cuda_diag()

    RUN_DIR.mkdir(parents=True, exist_ok=True)  # (solo por si decides guardar luego)

    # Data
    X_tr, X_va, Y_tr, Y_va, mlb = load_data(DATA_PATH)
    pos_weight = compute_pos_weight(Y_tr)

    # Tokenizer & Datasets
    tokenizer = prepare_tokenizer()
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    ds_tr = MovieDataset(X_tr, Y_tr, tokenizer, MAX_LEN)
    ds_va = MovieDataset(X_va, Y_va, tokenizer, MAX_LEN)

    # Model
    model = build_model(num_labels=Y_tr.shape[1], tokenizer=tokenizer)

    # Training
    lr = LR_LORA if USE_LORA else LR_FULL
    print_section(f"🏋️ TRAINING (Loss: {LOSS_TYPE})")
    args = TrainingArguments(
        output_dir=str(RUN_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        fp16=True,
        dataloader_pin_memory=True,
        report_to="none",
        seed=SEED,
    )

    trainer = CustomLossTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_default,
        loss_type=LOSS_TYPE,
        pos_weight=pos_weight if LOSS_TYPE == "bce" else None,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # Eval (threshold=0.5)
    print_section("📊 EVALUATION (threshold=0.5)")
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.6f}")

    # Threshold optimization
    preds = trainer.predict(ds_va)
    logits_va = preds.predictions
    thresholds = optimize_thresholds(logits_va, Y_va)
    tuned = evaluate_with_thresholds(logits_va, Y_va, thresholds)

    print_section("📈 EVALUATION (optimized thresholds)")
    for k, v in tuned.items():
        print(f"{k}: {v:.6f}")

    # ---------- imprime compute_metrics(y_va, pred) (si tienes validador externo) ----------
    probs_va = 1.0 / (1.0 + np.exp(-logits_va))
    preds_bin = (probs_va >= thresholds.reshape(1, -1)).astype(int)
    y_va = Y_va
    pred = preds_bin

    from validator import compute_metrics
    print(compute_metrics(y_va, pred))
    # ---------------------------------------------------------------------------------------

    # 🔗 Prepara el predictor EN MEMORIA (sin guardar ni cargar)
    prepare_predictor(model=trainer.model, tokenizer=tokenizer, thresholds=thresholds, labels=mlb.classes_.tolist())

    # 💡 Ejemplo de uso inmediato (sin tocar disco):
    df_test = pd.read_csv("../dataset_test.csv")
    out_df, pred_bin_test, probs_test = predict_in_memory(df_test, return_probs=True)
    print(out_df.head())

    print("\n✅ Done!")
    print(f"Artifacts (opcionales): {RUN_DIR.resolve()}")
    print(f"\n🎯 Best macro_f1: {tuned['macro_f1']:.4f}")

if __name__ == "__main__":
    main()