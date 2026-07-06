class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))
    
    def decode(self, ids):
        return bytes(ids).decode("utf-8")