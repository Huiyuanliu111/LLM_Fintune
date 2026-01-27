"""
分析添加的 256 个新 token (<0> ~ <255>) 的 embedding 相似度
包括：
1. 新 token 之间的相似度
2. 新 token 与现有 token 的相似度（数字、特殊符号等）
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
from pathlib import Path

# 配置
MODEL_NAME = "Qwen/Qwen3-0.6B"
# 如果想分析微调后的模型，可以改为:
# MODEL_NAME = "Qwen-Motion-Finetuned-v2/best_model"

def load_model_and_tokenizer(model_path: str, add_new_tokens: bool = True):
    """加载模型和 tokenizer"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    if add_new_tokens:
        # 添加新 tokens
        print("Adding new tokens <0>...<255>...")
        new_tokens = [f"<{i}>" for i in range(256)]
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} new tokens")
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU 足够分析 embedding
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    if add_new_tokens:
        # 调整 embedding 大小
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer, original_vocab_size

def get_new_token_embeddings(model, tokenizer, start_idx: int = 0, end_idx: int = 256):
    """获取新 token 的 embedding 向量"""
    # 获取 embedding 层
    embeddings = model.get_input_embeddings()
    
    # 获取新 token 的 ID
    new_token_ids = []
    for i in range(start_idx, end_idx):
        token = f"<{i}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_token_ids.append(token_id)
    
    # 提取 embedding 向量
    with torch.no_grad():
        token_embeddings = embeddings.weight[new_token_ids].float().numpy()
    
    return token_embeddings, new_token_ids

def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵"""
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    # 计算相似度矩阵
    similarity = np.dot(normalized, normalized.T)
    
    return similarity

def analyze_similarity(similarity_matrix: np.ndarray):
    """分析相似度统计信息"""
    n = similarity_matrix.shape[0]
    
    # 提取上三角（不包含对角线）
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    
    print("\n=== Similarity Statistics ===")
    print(f"Mean similarity: {upper_tri.mean():.4f}")
    print(f"Std similarity: {upper_tri.std():.4f}")
    print(f"Min similarity: {upper_tri.min():.4f}")
    print(f"Max similarity: {upper_tri.max():.4f}")
    print(f"Median similarity: {np.median(upper_tri):.4f}")
    
    # 找到最相似和最不相似的 token 对
    sim_copy = similarity_matrix.copy()
    np.fill_diagonal(sim_copy, -np.inf)
    
    max_idx = np.unravel_index(sim_copy.argmax(), sim_copy.shape)
    print(f"\nMost similar pair: <{max_idx[0]}> and <{max_idx[1]}> (sim={similarity_matrix[max_idx]:.4f})")
    
    np.fill_diagonal(sim_copy, np.inf)
    min_idx = np.unravel_index(sim_copy.argmin(), sim_copy.shape)
    print(f"Least similar pair: <{min_idx[0]}> and <{min_idx[1]}> (sim={similarity_matrix[min_idx]:.4f})")
    
    return upper_tri

def plot_similarity_matrix(similarity_matrix: np.ndarray, save_path: str = "token_similarity.png"):
    """绘制相似度热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 完整热力图
    ax1 = axes[0]
    im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title("Token Similarity Matrix (256 x 256)", fontsize=12)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("Token Index")
    plt.colorbar(im1, ax=ax1, label="Cosine Similarity")
    
    # 放大查看前 32 个 token
    ax2 = axes[1]
    subset = similarity_matrix[:32, :32]
    im2 = ax2.imshow(subset, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title("Zoom: First 32 Tokens", fontsize=12)
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("Token Index")
    ax2.set_xticks(range(0, 32, 4))
    ax2.set_yticks(range(0, 32, 4))
    plt.colorbar(im2, ax=ax2, label="Cosine Similarity")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved similarity matrix plot to: {save_path}")
    plt.show()

def plot_similarity_histogram(similarities: np.ndarray, save_path: str = "token_similarity_hist.png"):
    """绘制相似度分布直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=similarities.mean(), color='red', linestyle='--', label=f'Mean: {similarities.mean():.4f}')
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Pairwise Token Similarities")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram to: {save_path}")
    plt.show()

def analyze_embedding_norms(embeddings: np.ndarray):
    """分析 embedding 向量的范数"""
    norms = np.linalg.norm(embeddings, axis=1)
    
    print("\n=== Embedding Norm Statistics ===")
    print(f"Mean norm: {norms.mean():.4f}")
    print(f"Std norm: {norms.std():.4f}")
    print(f"Min norm: {norms.min():.4f} (token <{norms.argmin()}>)")
    print(f"Max norm: {norms.max():.4f} (token <{norms.argmax()}>)")
    
    return norms


def get_reference_token_embeddings(model, tokenizer, original_vocab_size: int):
    """获取一些有代表性的现有 token 的 embedding，用于对比分析"""
    embeddings_layer = model.get_input_embeddings()
    
    # 收集不同类型的参考 token
    reference_tokens = {}
    
    vocab = tokenizer.get_vocab()
    
    # 1. 数字 token (0-9, 以及多位数)
    print("\n--- Finding reference tokens ---")
    
    # 单个数字
    for digit in range(10):
        for fmt in [str(digit), f" {digit}", f"Ġ{digit}"]:  # 不同可能的格式
            if fmt in vocab:
                reference_tokens[f"digit_{digit}"] = (fmt, vocab[fmt])
                break
    
    # 两位数 (10, 20, ..., 90, 100, 200, 255)
    for num in [10, 20, 50, 100, 128, 200, 255]:
        for fmt in [str(num), f" {num}", f"Ġ{num}"]:
            if fmt in vocab:
                reference_tokens[f"num_{num}"] = (fmt, vocab[fmt])
                break
    
    # 2. 特殊符号
    special_chars = ['<', '>', '[', ']', '(', ')', '{', '}', '|', '/', '\\', '+', '-', '*', '=']
    for char in special_chars:
        if char in vocab:
            reference_tokens[f"char_{char}"] = (char, vocab[char])
    
    # 3. 常见词（包括动作相关词汇）
    common_words = ['the', 'and', 'is', 'motion', 'walk', 'run', 'jump', 'left', 'right',
                    'forward', 'backward', 'turn', 'step', 'stand', 'sit', 'kneel', 
                    'arm', 'leg', 'head', 'body', 'hand', 'foot', 'person', 'human']
    for word in common_words:
        for fmt in [word, f" {word}", f"Ġ{word}", word.capitalize(), f" {word.capitalize()}"]:
            if fmt in vocab:
                reference_tokens[f"word_{word}"] = (fmt, vocab[fmt])
                break
    
    # 4. 字节 token (如 <0x00>)
    for i in range(0, 256, 32):
        byte_token = f"<0x{i:02X}>"
        if byte_token in vocab:
            reference_tokens[f"byte_{i}"] = (byte_token, vocab[byte_token])
    
    # 5. 特殊 token
    for special in ['<|endoftext|>', '<|im_start|>', '<|im_end|>']:
        if special in vocab:
            reference_tokens[f"special_{special}"] = (special, vocab[special])
    
    print(f"Found {len(reference_tokens)} reference tokens")
    
    # 提取 embeddings
    ref_names = []
    ref_ids = []
    ref_tokens = []
    
    for name, (token, token_id) in reference_tokens.items():
        ref_names.append(name)
        ref_tokens.append(token)
        ref_ids.append(token_id)
    
    with torch.no_grad():
        ref_embeddings = embeddings_layer.weight[ref_ids].float().numpy()
    
    return ref_names, ref_tokens, ref_ids, ref_embeddings


def analyze_new_vs_existing(new_embeddings: np.ndarray, ref_embeddings: np.ndarray, 
                             ref_names: list, ref_tokens: list):
    """分析新 token 与现有参考 token 的相似度"""
    
    # 归一化
    new_norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    
    new_normalized = new_embeddings / (new_norms + 1e-8)
    ref_normalized = ref_embeddings / (ref_norms + 1e-8)
    
    # 计算交叉相似度矩阵: (256 x num_ref)
    cross_similarity = np.dot(new_normalized, ref_normalized.T)
    
    print("\n" + "="*60)
    print("NEW TOKENS vs EXISTING TOKENS SIMILARITY")
    print("="*60)
    
    # 对每个参考 token，找到最相似的新 token
    print("\n--- Most similar new token for each reference token ---")
    for i, (name, token) in enumerate(zip(ref_names, ref_tokens)):
        most_similar_new = np.argmax(cross_similarity[:, i])
        sim_value = cross_similarity[most_similar_new, i]
        print(f"  {name:20} ({repr(token):15}) -> <{most_similar_new}> (sim={sim_value:.4f})")
    
    # 对每个新 token (抽样)，找到最相似的参考 token
    print("\n--- Most similar reference token for selected new tokens ---")
    for new_idx in [0, 1, 10, 50, 100, 128, 200, 255]:
        most_similar_ref = np.argmax(cross_similarity[new_idx, :])
        sim_value = cross_similarity[new_idx, most_similar_ref]
        ref_name = ref_names[most_similar_ref]
        ref_token = ref_tokens[most_similar_ref]
        print(f"  <{new_idx:3}> -> {ref_name:20} ({repr(ref_token):15}) (sim={sim_value:.4f})")
    
    # 统计
    print("\n--- Cross-similarity Statistics ---")
    print(f"Mean: {cross_similarity.mean():.4f}")
    print(f"Std: {cross_similarity.std():.4f}")
    print(f"Max: {cross_similarity.max():.4f}")
    print(f"Min: {cross_similarity.min():.4f}")
    
    return cross_similarity


def plot_cross_similarity(cross_similarity: np.ndarray, ref_names: list, 
                           save_path: str = "cross_token_similarity.png"):
    """绘制新 token 与参考 token 的交叉相似度"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(cross_similarity.T, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xlabel("New Token Index (<0> to <255>)")
    ax.set_ylabel("Reference Token")
    ax.set_title("Cross-Similarity: New Tokens vs Existing Tokens")
    
    # Y 轴标签
    ax.set_yticks(range(len(ref_names)))
    ax.set_yticklabels(ref_names, fontsize=8)
    
    # X 轴标签（每 32 个标一次）
    ax.set_xticks(range(0, 256, 32))
    ax.set_xticklabels([f"<{i}>" for i in range(0, 256, 32)])
    
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved cross-similarity plot to: {save_path}")
    plt.show()


def analyze_motion_words_similarity(model, tokenizer, new_token_embeddings: np.ndarray):
    """
    专门分析新 token 与动作相关词汇的相似度
    这可以帮助理解训练后模型是否将 motion token 与动作语义联系起来
    """
    embeddings_layer = model.get_input_embeddings()
    vocab = tokenizer.get_vocab()
    
    # 动作相关词汇列表
    motion_words = [
        # 基本动作
        'walk', 'walking', 'run', 'running', 'jump', 'jumping',
        'stand', 'standing', 'sit', 'sitting', 'kneel', 'kneeling',
        'turn', 'turning', 'step', 'stepping',
        # 方向
        'forward', 'backward', 'left', 'right', 'up', 'down',
        'clockwise', 'circle',
        # 身体部位
        'arm', 'arms', 'leg', 'legs', 'hand', 'hands', 'foot', 'feet',
        'head', 'body', 'shoulder', 'knee', 'elbow',
        # 主体
        'person', 'human', 'man', 'woman', 'someone',
        # 动作描述
        'slowly', 'quickly', 'fast', 'slow',
        'motion', 'movement', 'action', 'gesture',
    ]
    
    # 查找词汇在词表中的 token
    found_words = {}
    for word in motion_words:
        # 尝试不同的格式
        for fmt in [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]:
            if fmt in vocab:
                found_words[word] = (fmt, vocab[fmt])
                break
    
    print("\n" + "="*60)
    print("MOTION WORDS SIMILARITY ANALYSIS")
    print("="*60)
    print(f"Found {len(found_words)}/{len(motion_words)} motion-related words in vocabulary")
    
    if len(found_words) == 0:
        print("No motion words found in vocabulary!")
        return None
    
    # 提取动作词汇的 embeddings
    word_names = list(found_words.keys())
    word_tokens = [found_words[w][0] for w in word_names]
    word_ids = [found_words[w][1] for w in word_names]
    
    with torch.no_grad():
        word_embeddings = embeddings_layer.weight[word_ids].float().numpy()
    
    # 计算相似度
    new_norms = np.linalg.norm(new_token_embeddings, axis=1, keepdims=True)
    word_norms = np.linalg.norm(word_embeddings, axis=1, keepdims=True)
    
    new_normalized = new_token_embeddings / (new_norms + 1e-8)
    word_normalized = word_embeddings / (word_norms + 1e-8)
    
    # 相似度矩阵: (256 x num_words)
    similarity = np.dot(new_normalized, word_normalized.T)
    
    # 打印每个动作词汇与新 token 的相似度
    print("\n--- Similarity of motion words with new tokens ---")
    print(f"{'Word':<15} {'Token':<15} {'Mean Sim':<10} {'Max Sim':<10} {'Max Token':<10}")
    print("-" * 60)
    
    for i, (word, token) in enumerate(zip(word_names, word_tokens)):
        sims = similarity[:, i]
        mean_sim = sims.mean()
        max_sim = sims.max()
        max_token_idx = sims.argmax()
        print(f"{word:<15} {repr(token):<15} {mean_sim:>8.4f}   {max_sim:>8.4f}   <{max_token_idx}>")
    
    # 找出与每个新 token 最相似的动作词汇
    print("\n--- Most similar motion word for each new token (sample) ---")
    for new_idx in [0, 32, 64, 96, 128, 160, 192, 224, 255]:
        most_similar_word_idx = np.argmax(similarity[new_idx, :])
        sim_value = similarity[new_idx, most_similar_word_idx]
        word = word_names[most_similar_word_idx]
        print(f"  <{new_idx:3}> -> {word:<15} (sim={sim_value:.4f})")
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(similarity.T, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xlabel("New Token Index (<0> to <255>)")
    ax.set_ylabel("Motion Word")
    ax.set_title("Similarity: New Tokens vs Motion-Related Words")
    
    ax.set_yticks(range(len(word_names)))
    ax.set_yticklabels(word_names, fontsize=8)
    
    ax.set_xticks(range(0, 256, 32))
    ax.set_xticklabels([f"<{i}>" for i in range(0, 256, 32)])
    
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    
    save_path = "motion_words_similarity.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved motion words similarity plot to: {save_path}")
    plt.show()
    
    return similarity, word_names


def main():
    # 检查是否有微调后的模型
    finetuned_path = Path("Qwen-Motion-Finetuned-v2/best_model")
    if finetuned_path.exists():
        print("Found finetuned model, will analyze both base and finetuned...")
        analyze_both = True
    else:
        print("Only analyzing base model (no finetuned model found)")
        analyze_both = False
    
    # 分析基础模型
    print("\n" + "="*60)
    print("ANALYZING BASE MODEL")
    print("="*60)
    
    model, tokenizer, original_vocab_size = load_model_and_tokenizer(MODEL_NAME)
    embeddings, token_ids = get_new_token_embeddings(model, tokenizer)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Token ID range: {min(token_ids)} to {max(token_ids)}")
    print(f"Original vocab size: {original_vocab_size}")
    
    # 分析范数
    norms = analyze_embedding_norms(embeddings)
    
    # 计算新 token 之间的相似度
    print("\n" + "="*60)
    print("NEW TOKENS MUTUAL SIMILARITY")
    print("="*60)
    similarity = compute_cosine_similarity(embeddings)
    upper_tri = analyze_similarity(similarity)
    
    # 绘图：新 token 之间的相似度
    plot_similarity_matrix(similarity, "base_token_similarity.png")
    plot_similarity_histogram(upper_tri, "base_token_similarity_hist.png")
    
    # 分析新 token 与现有 token 的相似度
    print("\n" + "="*60)
    print("ANALYZING NEW TOKENS vs EXISTING TOKENS")
    print("="*60)
    ref_names, ref_tokens, ref_ids, ref_embeddings = get_reference_token_embeddings(
        model, tokenizer, original_vocab_size
    )
    
    cross_sim = analyze_new_vs_existing(embeddings, ref_embeddings, ref_names, ref_tokens)
    plot_cross_similarity(cross_sim, ref_names, "base_cross_similarity.png")
    
    # 分析新 token 与动作词汇的相似度
    analyze_motion_words_similarity(model, tokenizer, embeddings)
    
    # 保存 embeddings 用于后续重新加载
    base_embeddings = embeddings.copy()
    
    # 清理内存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 如果有微调后的模型，也分析它
    if analyze_both:
        print("\n" + "="*60)
        print("ANALYZING FINETUNED MODEL")
        print("="*60)
        
        model_ft, tokenizer_ft, _ = load_model_and_tokenizer(str(finetuned_path))
        embeddings_ft, _ = get_new_token_embeddings(model_ft, tokenizer_ft)
        
        print(f"\nEmbedding shape: {embeddings_ft.shape}")
        
        norms_ft = analyze_embedding_norms(embeddings_ft)
        similarity_ft = compute_cosine_similarity(embeddings_ft)
        upper_tri_ft = analyze_similarity(similarity_ft)
        
        plot_similarity_matrix(similarity_ft, "finetuned_token_similarity.png")
        plot_similarity_histogram(upper_tri_ft, "finetuned_token_similarity_hist.png")
        
        # 分析微调后的新 token 与现有 token 的关系
        ref_names_ft, ref_tokens_ft, ref_ids_ft, ref_embeddings_ft = get_reference_token_embeddings(
            model_ft, tokenizer_ft, original_vocab_size
        )
        cross_sim_ft = analyze_new_vs_existing(embeddings_ft, ref_embeddings_ft, ref_names_ft, ref_tokens_ft)
        plot_cross_similarity(cross_sim_ft, ref_names_ft, "finetuned_cross_similarity.png")
        
        # 分析微调后的新 token 与动作词汇的相似度
        analyze_motion_words_similarity(model_ft, tokenizer_ft, embeddings_ft)
        
        # 比较基础模型和微调模型
        print("\n" + "="*60)
        print("COMPARISON: BASE vs FINETUNED")
        print("="*60)
        
        # 计算每个 token 的 embedding 变化
        embedding_diff = np.linalg.norm(embeddings_ft - base_embeddings, axis=1)
        print(f"Mean embedding change (L2): {embedding_diff.mean():.4f}")
        print(f"Max embedding change: {embedding_diff.max():.4f} (token <{embedding_diff.argmax()}>)")
        print(f"Min embedding change: {embedding_diff.min():.4f} (token <{embedding_diff.argmin()}>)")
        
        # 打印变化最大的 10 个 token
        top_changed = np.argsort(embedding_diff)[-10:][::-1]
        print("\nTop 10 most changed tokens:")
        for idx in top_changed:
            print(f"  <{idx}>: L2 change = {embedding_diff[idx]:.4f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

