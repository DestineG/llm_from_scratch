# src/utils/link_data.py

import os
import subprocess
import platform
import argparse

def link_dir(src, dst, force=False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst):
        if force:
            print(f"[remove] existing path: {dst}")
            if os.path.islink(dst) or os.path.isfile(dst):
                os.remove(dst)
            else:
                os.rmdir(dst)
        else:
            print(f"[skip] {dst} already exists (use --force to override)")
            return

    system = platform.system()

    if system == "Windows":
        try:
            os.symlink(src, dst, target_is_directory=True)
            print(f"[symlink] {dst} -> {src}")
        except OSError:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", dst, src],
                check=True
            )
            print(f"[junction] {dst} -> {src}")
    else:
        os.symlink(src, dst)
        print(f"[symlink] {dst} -> {src}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a symbolic link (or junction on Windows) for dataset directories"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="真实数据所在路径（例如：D:/LLM_DATA/tiny_shakespeare）"
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="项目内目标路径（例如：data/raw/tiny_shakespeare）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果目标路径已存在，强制删除并重新创建"
    )

    args = parser.parse_args()

    link_dir(
        src=os.path.abspath(args.src),
        dst=os.path.abspath(args.dst),
        force=args.force
    )


# python -m src.utils.link_data --src /dataroot/liujiang/data/datasets/wmt_zh_en_training_corpus.csv --dst data/raw/wmt_zh_en_training_corpus/wmt_zh_en_training_corpus.csv
if __name__ == "__main__":
    main()
