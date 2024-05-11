import sys

from yigpt.tokenization.bbpe import BBPETokenizer

if __name__ == "__main__":
    txt_fn = sys.argv[1]
    vocab_size = int(sys.argv[2])
    out = sys.argv[3]

    with open(txt_fn, encoding="utf-8") as f:
        txt = f.read()

    tokenizer = BBPETokenizer()
    tokenizer.train(txt, vocab_size,verbose=True)
    tokenizer.save(out)
