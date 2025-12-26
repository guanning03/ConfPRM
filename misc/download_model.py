#!/usr/bin/env python3
"""
优化的模型下载脚本
支持多种下载方式和镜像加速
"""

import os
import sys
import time
from typing import Optional

def setup_environment(cache_dir: Optional[str] = None, use_mirror: bool = True):
    """设置下载环境"""
    # 设置缓存目录
    CACHE = os.path.expanduser(cache_dir or os.environ.get("CACHE", "~/.cache"))
    assert CACHE, "$CACHE is empty"
    
    hf_home = os.path.join(CACHE, "hf_home")
    hf_hub_cache = os.path.join(hf_home, "hub")
    tfm_cache = os.path.join(hf_home, "transformers")
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(tfm_cache, exist_ok=True)
    
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = tfm_cache
    
    # 设置镜像（如果需要）
    if use_mirror:
        # 优先使用HF-Mirror（国内镜像）
        mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = mirror
        print(f"✅ 使用镜像: {mirror}")
    
    # 检查代理设置
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
    has_proxy = any(os.environ.get(var) for var in proxy_vars)
    if has_proxy:
        print("✅ 检测到代理设置")
    else:
        print("ℹ️  未设置代理，如果下载慢可考虑配置代理")
    
    return CACHE, hf_hub_cache

def download_with_huggingface_hub(model_name: str, cache_dir: str, local_path: str, 
                                   use_mirror: bool = True, resume: bool = True):
    """使用huggingface_hub下载"""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\n📥 方法1: 使用 huggingface_hub.snapshot_download")
        print(f"   模型: {model_name}")
        print(f"   目标路径: {local_path}")
        
        start_time = time.time()
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            cache_dir=cache_dir,
            resume_download=resume,
            # 使用多线程下载
            max_workers=4,
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 下载完成！耗时: {elapsed:.2f} 秒")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_with_huggingface_cli(model_name: str, local_path: str):
    """使用huggingface-cli命令行工具下载"""
    try:
        import subprocess
        
        print(f"\n📥 方法2: 使用 huggingface-cli download")
        print(f"   模型: {model_name}")
        print(f"   目标路径: {local_path}")
        
        start_time = time.time()
        
        cmd = [
            "huggingface-cli", "download",
            model_name,
            "--local-dir", local_path,
            "--local-dir-use-symlinks", "False",
            "--resume-download"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✅ 下载完成！耗时: {elapsed:.2f} 秒")
            return True
        else:
            print(f"❌ 下载失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ huggingface-cli 未安装，跳过此方法")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_with_modelscope(model_name: str, local_path: str):
    """使用ModelScope镜像下载"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        
        print(f"\n📥 方法3: 使用 ModelScope 镜像")
        print(f"   模型: {model_name}")
        print(f"   目标路径: {local_path}")
        
        start_time = time.time()
        
        snapshot_download(
            model_id=model_name,
            cache_dir=local_path,
            resume_download=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 下载完成！耗时: {elapsed:.2f} 秒")
        return True
        
    except ImportError:
        print("❌ modelscope 未安装，跳过此方法")
        print("   安装命令: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_with_git_lfs(model_name: str, local_path: str):
    """使用git-lfs下载（适合大文件）"""
    try:
        import subprocess
        
        print(f"\n📥 方法4: 使用 git-lfs")
        print(f"   模型: {model_name}")
        print(f"   目标路径: {local_path}")
        
        # 构建HuggingFace仓库URL
        repo_url = f"https://huggingface.co/{model_name}"
        
        # 检查是否有镜像
        mirror = os.environ.get("HF_ENDPOINT", "")
        if mirror and "hf-mirror.com" in mirror:
            repo_url = f"https://hf-mirror.com/{model_name}"
        
        start_time = time.time()
        
        # 克隆仓库
        cmd = ["git", "clone", repo_url, local_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 拉取LFS文件
            os.chdir(local_path)
            lfs_cmd = ["git", "lfs", "pull"]
            result = subprocess.run(lfs_cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            if result.returncode == 0:
                print(f"✅ 下载完成！耗时: {elapsed:.2f} 秒")
                return True
            else:
                print(f"⚠️  Git克隆成功，但LFS拉取失败: {result.stderr}")
                return False
        else:
            print(f"❌ Git克隆失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ git 或 git-lfs 未安装，跳过此方法")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="优化的模型下载工具")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B",
                       help="模型名称 (默认: Qwen/Qwen2.5-Math-7B)")
    parser.add_argument("--cache", type=str, default=None,
                       help="缓存目录 (默认: $CACHE 或 ~/.cache)")
    parser.add_argument("--method", type=str, 
                       choices=["auto", "hub", "cli", "modelscope", "git"],
                       default="auto",
                       help="下载方法 (默认: auto，自动选择最快的方法)")
    parser.add_argument("--no-mirror", action="store_true",
                       help="不使用镜像")
    parser.add_argument("--try-all", action="store_true",
                       help="尝试所有方法直到成功")
    
    args = parser.parse_args()
    
    print("="*60)
    print("优化的模型下载工具")
    print("="*60)
    
    # 设置环境
    cache_dir, hf_hub_cache = setup_environment(args.cache, use_mirror=not args.no_mirror)
    
    # 准备本地路径
    local_path = os.path.join(cache_dir, f"hf_models/{args.model}")
    os.makedirs(local_path, exist_ok=True)
    
    # 检查是否已存在
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"\n⚠️  目录已存在且非空: {local_path}")
        response = input("是否继续下载？(y/n): ")
        if response.lower() != 'y':
            print("已取消下载")
            return
    
    # 选择下载方法
    methods = []
    if args.method == "auto" or args.method == "hub":
        methods.append(("hub", lambda: download_with_huggingface_hub(
            args.model, hf_hub_cache, local_path, use_mirror=not args.no_mirror)))
    
    if args.method == "auto" or args.method == "cli":
        methods.append(("cli", lambda: download_with_huggingface_cli(args.model, local_path)))
    
    if args.method == "auto" or args.method == "modelscope":
        methods.append(("modelscope", lambda: download_with_modelscope(args.model, local_path)))
    
    if args.method == "auto" or args.method == "git":
        methods.append(("git", lambda: download_with_git_lfs(args.model, local_path)))
    
    # 执行下载
    if args.try_all:
        # 尝试所有方法直到成功
        for method_name, method_func in methods:
            print(f"\n尝试方法: {method_name}")
            if method_func():
                print(f"\n✅ 使用 {method_name} 方法下载成功！")
                print(f"📁 文件保存在: {local_path}")
                return
        print("\n❌ 所有方法都失败了")
    else:
        # 只尝试第一个方法
        if methods:
            method_name, method_func = methods[0]
            if method_func():
                print(f"\n✅ 下载成功！")
                print(f"📁 文件保存在: {local_path}")
            else:
                print(f"\n❌ 下载失败，可以尝试:")
                print(f"   1. 使用 --try-all 参数尝试所有方法")
                print(f"   2. 使用 --method 指定其他方法")
                print(f"   3. 检查网络连接和代理设置")
        else:
            print("❌ 没有可用的下载方法")

if __name__ == "__main__":
    main()
