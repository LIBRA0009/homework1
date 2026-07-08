_base_ = [
    './segformer_mit-b2_8xb8-17k_suim-dgmf-512x512.py',
]

model = dict(decode_head=dict(residual_scale=0.05))
