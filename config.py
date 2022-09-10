class Config:
    WARMUP_STEPS = 20
    MAX_SEQ_LEN = 10
    EMBED_DIM = 512
    HIDDEN_DIM = 2048
    N_HEADS = 8
    ENCODER_N_BLOCKS = 6
    DECODER_N_BLOCKS = 6
    ENG_VOCAB_SIZE = -1
    FR_VOCAB_SIZE = -1
    BATCH_SIZE = 2
    DEVICE = "cuda"