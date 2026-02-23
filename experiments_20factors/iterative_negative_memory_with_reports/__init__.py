# __init__ 里加一行：
self.batch_size = max(20, int(pos_cfg.get('batch_size', 20)))
