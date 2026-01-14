# src/train/trainer.py

import os
import math
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import build_dataloader_from_owt_en_bpeTokenizer
from src.model.gpt import GPT

@dataclass
class TrainConfig:
    # 实验设置
    exp_name: str = "gpt_owt_en_bpeTokenizer_with_warmup_v2"
    exps_dir: str = "./experiments"
    
    # 训练超参数
    epochs: int = 1
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    
    # 间隔设置
    save_steps: int = 100000
    log_steps: int = 500
    
    # 模型配置
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "embed_dim": 768,
        "seq_len": 128,
        "num_heads": 12,
        "ff_dim": 3072,
        "num_layers": 8,
    })

    def __post_init__(self):
        self.exp_dir = os.path.abspath(os.path.join(self.exps_dir, self.exp_name))
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.log_path = os.path.join(self.exp_dir, "training_log.txt")


class Trainer:
    def __init__(self, config: TrainConfig, device: torch.device):
        self.config = config
        self.device = device
        self.global_step = 0
        
        # 初始化工作目录
        self._prepare_dirs()
        
        # 加载数据与分词器
        self.dataloader, self.tokenizer = self._setup_data()
        
        # 更新并初始化模型
        self.config.model_params["vocab_size"] = self.tokenizer.n_vocab
        self.model = GPT(**self.config.model_params).to(self.device)
        
        # 优化器与调度器
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        self.scheduler = self._get_scheduler()
        
        # 记录器
        self.writer = SummaryWriter(os.path.join(self.config.exp_dir, 'tensorboard_logs'))
        self._log_experiment_config()

    def _prepare_dirs(self):
        if os.path.exists(self.config.exp_dir):
            shutil.rmtree(self.config.exp_dir)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _setup_data(self):
        return build_dataloader_from_owt_en_bpeTokenizer(
            text_nums=10000,
            window_size=self.config.model_params["seq_len"],
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=True,
            bpe_model_name="gpt2"
        )

    def _get_scheduler(self):
        total_steps = len(self.dataloader) * self.config.epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step >= total_steps:
                return 0.0
            
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            min_lr_ratio = 0.01
            cosine_decay = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            return max(0.0, cosine_decay)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _log_experiment_config(self):
        with open(self.config.log_path, "w") as f:
            f.write("========== Experiment Config ==========\n")
            f.write(f"Experiment: {self.config.exp_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total Epochs: {self.config.epochs}\n\n")
            f.write("Model Config:\n")
            for k, v in self.config.model_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nTokenizer: {self.tokenizer.customName}\n")
            f.write("=======================================\n\n")

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            
            batch_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch [{epoch+1}/{self.config.epochs}]", 
                ascii=True, leave=False
            )

            for batch in batch_bar:
                inputs, targets = [get.to(self.device) for get in batch]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1
                total_loss += loss.item()
                
                # 更新进度条
                batch_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

                # 定期记录
                if self.global_step % self.config.log_steps == 0:
                    self._record_metrics(loss.item(), inputs[0], outputs[0], targets[0])

                # 定期保存
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

        # 训练结束，保存最终模型
        self._save_final_model()

    def _record_metrics(self, loss, sample_in, sample_out, sample_target):
        # 写入文本日志
        with open(self.config.log_path, "a") as f:
            f.write(f"Step {self.global_step}, Loss: {loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}\n"
                    f"In:  {self.tokenizer.decode(sample_in.cpu().tolist())[:100]}...\n"
                    f"Target: {self.tokenizer.decode(sample_target.cpu().tolist())[:100]}...\n"
                    f"Out: {self.tokenizer.decode(torch.argmax(sample_out, dim=-1).cpu().tolist())[:100]}...\n\n")
        
        # Tensorboard
        self.writer.add_scalar('Loss/train', loss, self.global_step)
        self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], self.global_step)

    def _save_checkpoint(self):
        path = os.path.join(self.config.checkpoint_dir, f"model_step_{self.global_step}.pt")
        torch.save(self.model.state_dict(), path)
    
    def _save_final_model(self):
        path = os.path.join(self.config.checkpoint_dir, "model_final.pt")
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig()
    trainer = Trainer(config, device)
    trainer.train()