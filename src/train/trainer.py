# src/train/trainer.py

import os
import shutil
from tqdm import tqdm
import torch
from src.data.dataset import build_dataloader_from_wmt_en_basicTokenizer, build_dataloader_from_wmt_en_bpeTokenizer
from src.model.gpt import GPT


def log_experiment_config(
    log_path,
    exp_name,
    device,
    epochs,
    model_config,
    tokenizer,
):
    with open(log_path, "w") as f:  # 用 "w"，确保是新实验
        f.write("========== Experiment Config ==========\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Save steps: {save_steps}\n")
        f.write(f"Log steps: {log_steps}\n\n")

        f.write("Model Config:\n")
        for k, v in model_config.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nTokenizer:\n")
        f.write(f"  type: {tokenizer.customName}\n")
        f.write(f"  vocab_size: {tokenizer.n_vocab}\n")

        f.write("=======================================\n\n")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    global_step,
    epoch,
    epochs,
    tokenizer,
):
    model.train()
    total_loss = 0.0

    batch_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch+1}/{epochs}]",
        ascii=True,
        leave=False,
    )

    for batch in batch_bar:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        batch_bar.set_postfix({
            "avg_loss": f"{total_loss / (batch_bar.n + 1):.4f}",
            "step": global_step
        })
        if global_step % log_steps == 0:
            os.makedirs(exp_dir, exist_ok=True)

            with open(log_path, "a") as f:
                f.write(
                    f"Step {global_step}, Loss: {loss.item():.4f}\n"
                    f"Inputs: {tokenizer.decode(inputs[0].cpu().numpy().tolist())}\n"
                    f"Outputs: {tokenizer.decode(torch.argmax(outputs[0], dim=-1).cpu().numpy().tolist())}\n"
                    f"Targets: {tokenizer.decode(targets[0].cpu().numpy().tolist())}\n\n"
                )

        if global_step % save_steps == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_step_{global_step}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, global_step

model_config = {
    "vocab_size": 50000,
    "embed_dim": 768,
    "seq_len": 128,
    "num_heads": 12,
    "ff_dim": 3072,
    "num_layers": 8,
}
save_steps = 10000
log_steps = 1000
exps_dir = "./experiments"
exp_name = "gpt_wmt_en_bpeTokenizer"
exp_dir = os.path.abspath(os.path.join(exps_dir, exp_name))
if os.path.exists(exp_dir):
    shutil.rmtree(exp_dir)   # 删除整个目录（包括所有文件和子目录）
os.makedirs(exp_dir)
checkpoint_dir = os.path.join(exp_dir, "checkpoints")
log_path = os.path.join(exp_dir, "training_log.txt")
def train(epochs, device):
    # dataloader, tokenizer = build_dataloader_from_wmt_en_basicTokenizer(
    #     seq_nums=1000000, vocab_size=model_config.get("vocab_size"),
    #     window_size=model_config.get("seq_len"), batch_size=32, num_workers=4, shuffle=True,
    # )
    dataloader, tokenizer = build_dataloader_from_wmt_en_bpeTokenizer(
        seq_nums=1000000, window_size=model_config.get("seq_len"), batch_size=32,
        num_workers=4, shuffle=True, bpe_model_name="gpt2"
    )

    model_config.update({"vocab_size": tokenizer.n_vocab})
    print(f"DataLoader built with {len(dataloader)} batches.")

    model = GPT(**model_config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.95
    )

    # 记录实验配置
    log_experiment_config(
        log_path,
        exp_name,
        device,
        epochs,
        model_config,
        tokenizer,
    )

    global_step = 0

    for epoch in range(epochs):
        avg_loss, global_step = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            global_step,
            epoch,
            epochs,
            tokenizer,
        )
        scheduler.step()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(epochs=1000, device=device)