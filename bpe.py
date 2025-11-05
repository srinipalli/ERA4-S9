from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
from tokenizers.normalizers import NFKC
from datasets import load_dataset
import math

# ---- Step 1: Load a Telugu dataset ----
# Using Wikipedia subset (~20MB) for realism
dataset = load_dataset("eswardivi/telugu_dataset", "telugu_nlp", split="train[:2%]")
texts = [x["text"] for x in dataset if x["text"].strip()]

# ---- Step 2: Initialize empty BPE tokenizer ----
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([NFKC()])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# ---- Step 3: Train the tokenizer ----
trainer = trainers.BpeTrainer(
    vocab_size=4800,  # <5000 as required
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train_from_iterator(texts, trainer)

# ---- Step 4: Save the model ----
tokenizer.save("telugu_bpe.json")

# ---- Step 5: Evaluate compression ratio ----
def compression_ratio(sample_texts):
    raw_bytes = sum(len(t.encode("utf-8")) for t in sample_texts)
    tokens = [len(tokenizer.encode(t).tokens) for t in sample_texts]
    avg_token_len = raw_bytes / sum(tokens)
    return avg_token_len

sample_texts = texts[:1000]
ratio = compression_ratio(sample_texts)
print(f"Compression ratio (bytes per token): {ratio:.2f}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# ---- Step 6: Quick Test ----
sample = "తెలంగాణ ప్రభుత్వం ఆర్థికాభివృద్ధి కోసం పథకాలు అమలు చేస్తోంది."
encoded = tokenizer.encode(sample)
print("\nOriginal:", sample)
print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
