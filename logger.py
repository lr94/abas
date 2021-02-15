import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, logdir: str = 'tensorboard', run_name=None, use_tb: bool = True, use_tqdm: bool = True,
                 *args, **kwargs):
        self.use_tb = use_tb
        self.use_tqdm = use_tqdm
        self.epoch = 0

        if use_tb:
            log_path = logdir if run_name is None else os.path.join(logdir, run_name)
            self.summary_writer = SummaryWriter(log_dir=log_path)

    def progress(self, *args, **kwargs):
        if self.use_tqdm:
            return tqdm(*args, **kwargs)
        else:
            # Alternative output if tqdm is not enabled
            class ProgressIndicator:
                def __init__(self, total: int, desc: str = "", *args, **kwargs):
                    self.total = total
                    self.desc = desc
                    self.counter = 0
                    self.old_line = ''

                def __enter__(self):
                    print("Starting {}".format(self.desc))
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    print("Done {}".format(self.desc))

                def update(self, n: int):
                    self.counter += n
                    line = "{} ({} %)".format(self.desc,
                                              int(self.counter / self.total * 100))
                    if line != self.old_line:
                        print(line)
                        self.old_line = line

            return ProgressIndicator(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        if self.use_tb:
            self.summary_writer.add_scalar(*args, **kwargs)
