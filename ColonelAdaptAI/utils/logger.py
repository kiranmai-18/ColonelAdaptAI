import os, json, time
from pathlib import Path

class ExperimentLogger:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.out_dir / 'log.txt'
    def log(self, msg):
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        text = f'[{stamp}] {msg}'
        print(text, flush=True)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    def save_json(self, obj, name):
        with open(self.out_dir / name, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)
