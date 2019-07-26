#! /usr/bin/env python3

import numpy as np

def stochastic_mini_batch(input_tensor, input_data, batch_size):
    idxs = np.random.choice(
        input_data.shape[0], size=batch_size, replace=False
    )
    return {input_tensor: input_data[idxs]}

def log_and_saver_setup(logdir, ckpt_id="model.ckpt"):
    from datetime import datetime
    import os

    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    outdir = f"{logdir}/run-{now}"
    ckpt = f"{outdir}/{ckpt_id}"
    return ckpt


if __name__ == "__main__":
    pass
