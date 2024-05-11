from yigpt.tokenization.bbpe import BBPETokenizer


if __name__ == "__main__":
    tokenizer = BBPETokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3)
    print(tokenizer.vocab)
    print(tokenizer.merges)

    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(tokenizer.encode(text)) == text

    tokens = tokenizer.tokenize(text)
    print(tokens)
