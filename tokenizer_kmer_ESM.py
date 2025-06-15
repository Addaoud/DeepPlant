from transformers import EsmTokenizer
from src.tokenizers import KmerEsmTokenizer
import itertools
from typing import List
from src.utils import create_path
import os
from typing import Optional


def generate_kmers(k=4):
    """Generate all possible k-mers from DNA alphabet A, C, G, T."""
    kmer_list = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    return kmer_list


def write_kmer_corpus(k=6, file_path="kmer_corpus.txt"):
    """
    Pre-tokenize sequences into k-mers and write them to a corpus file.
    """
    special_tokens = ["<unk>", "<pad>", "<mask>", "<cls>"]
    kmers = generate_kmers(k=k)
    pos_tokens = ["<eos>", "<bos>"]
    vocab = kmers + special_tokens + pos_tokens
    with open(file_path, "w") as f:
        for token in vocab:
            f.write(token + "\n")


def train_kmer_tokenizer(vocab_path, save_dir="./dna_tokenizer"):
    """
    Train a BPE tokenizer on a k-mer corpus using Hugging Face's Tokenizers.
    """
    tokenizer = KmerEsmTokenizer(
        vocab_file=vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<cls>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_length=500,
    )

    tokenizer.save_pretrained(save_dir)


# ---------------------
# Example Usage
# ---------------------
if __name__ == "__main__":
    # sequences = pd.read_csv(
    #    "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_200/2500_Seq_10kb_masked.csv"
    # ).sequence.tolist()

    k = 5
    tokenizer_save_dir = "tokenizers/ESM_kmer_tokenizer/"
    create_path(tokenizer_save_dir)

    # Step 1: Write k-mer corpus
    corpus_file = os.path.join(tokenizer_save_dir, f"{k}mer_corpus.txt")
    if not os.path.exists(corpus_file):
        print("writing corpus file")
        write_kmer_corpus(k=k, file_path=corpus_file)

    # Step 2: Train tokenizer
    train_kmer_tokenizer(vocab_path=corpus_file, save_dir=tokenizer_save_dir)

    # tokenizer = KmerEsmTokenizer.from_pretrained(tokenizer_dir)
    # Test
