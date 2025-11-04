import os
import git
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict

REPO_URL = "https://github.com/apache/avro.git"  # Avro 仓库地址
LOCAL_REPO_PATH = "./avro_repo"  # 本地仓库路径
JIRA_BUG_FILE = "./AVRO_Bugs_merged.xml"  # XML 文件
OUTPUT_CSV = "defect_prediction_dataset.csv"  # 输出包含标签的训练集
TARGET_VERSION = "release-1.7.0"  # 早期版本（时间节点，可修改）

# 原有函数：完全保留，不做任何修改
def clone_or_update_repo(repo_url, local_path):
    if not os.path.exists(local_path):
        print("Cloning repository... (这可能需要几分钟，取决于网络速度)")
        # 克隆完整历史（不加depth限制，确保能获取所有提交）
        repo = git.Repo.clone_from(repo_url, local_path, multi_options=["--recursive"])
    else:
        print("Updating repository...")
        repo = git.Repo(local_path)
        repo.remotes.origin.pull()  # 拉取最新代码
    return repo

def get_latest_files_all(repo):
    """获取最新版本的所有文件路径（统一转为小写，便于匹配）"""
    latest_commit = repo.head.commit
    tree = latest_commit.tree
    latest_files = []

    def traverse_tree(t):
        for blob in t:
            if blob.type == "blob":  # 只保留文件（排除目录）
                # 统一路径为小写，解决大小写敏感问题
                file_path = blob.path.lower()
                latest_files.append(file_path)
            elif blob.type == "tree":
                traverse_tree(blob)

    traverse_tree(tree)
    # 去重（避免重复路径）
    latest_files = list(set(latest_files))
    print(f"Latest version has {len(latest_files)} unique files (lowercase)")
    return latest_files

def extract_git_features(repo, latest_files):
    """提取Git变更特征（修复路径匹配，添加调试反馈）"""
    file_stats = defaultdict(lambda: {
        "ChangeRate": 0,
        "ChangeLOC": 0,
        "AddLOC": 0,
        "DeleteLOC": 0,
        "#Author": set(),
        "LOC": 0
    })

    # 调试：统计遍历的提交数
    commit_count = 0
    matched_file_count = 0  # 统计匹配到的文件变更次数

    print("Extracting Git features... (this may take a while)")
    # 遍历所有提交（从旧到新，reverse=True更符合直觉，不影响结果）
    for commit in repo.iter_commits(reverse=True):
        commit_count += 1
        author = commit.author.email
        stats = commit.stats.files  # 该提交的文件变更统计

        # 调试：每1000次提交打印进度
        if commit_count % 1000 == 0:
            print(f"Processed {commit_count} commits, matched {matched_file_count} file changes")

        if not stats:  # 跳过无文件变更的提交
            continue

        # 遍历该提交变更的所有文件
        for f, s in stats.items():
            # 统一路径为小写，与最新文件列表匹配
            f_lower = f.lower()
            if f_lower in latest_files:
                matched_file_count += 1
                file_stats[f_lower]["ChangeRate"] += 1  # 变更次数+1
                file_stats[f_lower]["ChangeLOC"] += s['lines']  # 总变更行数
                file_stats[f_lower]["AddLOC"] += s['insertions']  # 新增行数
                file_stats[f_lower]["DeleteLOC"] += s['deletions']  # 删除行数
                file_stats[f_lower]["#Author"].add(author)  # 记录作者

    # 打印调试信息：确认提交和匹配情况
    print(f"Total processed commits: {commit_count}")
    print(f"Total matched file changes: {matched_file_count}")
    print(f"Total files with Git features: {len([f for f in file_stats if file_stats[f]['ChangeRate'] > 0])}")

    # 计算最新版本的LOC（代码行数）
    print("Calculating LOC for latest files...")
    for f_lower in latest_files:
        # 还原为原始路径（因为最新文件列表是小写，本地文件路径可能大小写不同）
        # 这里通过遍历最新提交的文件树，找到原始路径
        original_path = None
        for blob in repo.head.commit.tree.traverse():
            if blob.type == "blob" and blob.path.lower() == f_lower:
                original_path = blob.path
                break

        if original_path:
            file_path = os.path.join(LOCAL_REPO_PATH, original_path)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                    file_stats[f_lower]["LOC"] = len(lines)
            except Exception as e:
                print(f"Failed to read {original_path}: {e}")
                file_stats[f_lower]["LOC"] = 0
        else:
            file_stats[f_lower]["LOC"] = 0

    # 转换#Author为数量（set转int）
    for f in file_stats:
        file_stats[f]["#Author"] = len(file_stats[f]["#Author"])

    return file_stats

def load_jira_bugs_xml(jira_file):
    """加载JIRA Bug数据（保持原逻辑，添加调试）"""
    if not os.path.exists(jira_file):
        print(f"Warning: JIRA file {jira_file} not found! Bug features will be 0.")
        return pd.DataFrame(columns=["BugID", "Type", "Priority"])

    tree = ET.parse(jira_file)
    root = tree.getroot()
    priority_map = {"Blocker": 5, "Critical": 4, "Major": 3, "Minor": 2, "Trivial": 1}
    bug_list = []

    for item in root.findall(".//item"):
        bug_id = item.findtext("key")
        bug_type = item.findtext("type")
        priority_text = item.findtext("priority")

        if bug_id and bug_type and priority_text:
            priority_text = priority_text.strip()
            priority_value = priority_map.get(priority_text, 1)  # 未知优先级设为1
            bug_list.append({
                "BugID": bug_id.strip(),
                "Type": bug_type.strip(),
                "Priority": priority_value
            })

    df = pd.DataFrame(bug_list)
    print(f"Loaded {len(df)} bugs from JIRA XML")
    return df

def extract_bug_features(repo, latest_files, jira_df):
    """提取Bug特征（修复路径匹配）"""
    if jira_df.empty:
        print("No JIRA bugs loaded. Bug features will be 0.")
        return {f: {"BugRate": 0, "AvgBugPriority": 0} for f in latest_files}

    bug_stats = {f: {"BugRate": 0, "AvgBugPriority": 0} for f in latest_files}
    bug_priority_map = dict(zip(jira_df['BugID'], jira_df['Priority']))
    commit_count = 0
    matched_bug_commit_count = 0  # 统计包含BugID的提交数

    print("Extracting Bug features...")
    for commit in repo.iter_commits(reverse=True):
        commit_count += 1
        msg = commit.message.lower()  # 统一为小写，避免大小写敏感

        # 匹配提交信息中的BugID
        matched_bugs = [bug for bug in bug_priority_map if bug.lower() in msg]
        if not matched_bugs:
            continue

        matched_bug_commit_count += 1
        stats = commit.stats.files
        if not stats:
            continue

        # 遍历该提交变更的文件，匹配最新文件列表（小写）
        for f, _ in stats.items():
            f_lower = f.lower()
            if f_lower in latest_files:
                for bug in matched_bugs:
                    bug_stats[f_lower]["BugRate"] += 1
                    bug_stats[f_lower]["AvgBugPriority"] += bug_priority_map[bug]

    # 打印调试信息
    print(f"Total processed commits for bugs: {commit_count}")
    print(f"Commits with BugID: {matched_bug_commit_count}")
    print(f"Total files with Bug features: {len([f for f in bug_stats if bug_stats[f]['BugRate'] > 0])}")

    # 计算平均优先级
    for f in bug_stats:
        if bug_stats[f]["BugRate"] > 0:
            bug_stats[f]["AvgBugPriority"] /= bug_stats[f]["BugRate"]

    return bug_stats

# 新增函数1：获取早期版本（时间节点）的commit
def get_target_version_commit(repo, target_version):
    """仅获取早期版本的commit，不修改原有逻辑"""
    try:
        tags = {tag.name.lower(): tag for tag in repo.tags}
        target_version_lower = target_version.lower()
        if target_version_lower not in tags:
            existing_tags = list(tags.keys())[:10]
            raise ValueError(f"未找到目标版本标签: {target_version}\n可用早期标签示例: {existing_tags}")
        target_tag = tags[target_version_lower]
        target_commit = target_tag.commit
        print(f"\n✅ 时间节点配置完成")
        print(f"目标早期版本: {target_version}")
        print(f"对应Commit: {target_commit.hexsha}")
        print(f"时间节点: {target_commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        return target_commit
    except Exception as e:
        print(f"获取时间节点失败: {e}")
        raise

# 新增函数2：标注两个标签（is_buggy/bug_count）
def add_defect_labels(repo, latest_files, jira_df, target_commit):
    """仅新增标签，不影响原有特征"""
    if jira_df.empty:
        print("⚠️  无JIRA数据，标签均设为0")
        return {f: {"is_buggy": 0, "bug_count": 0} for f in latest_files}

    labels = {f: {"is_buggy": 0, "bug_count": 0} for f in latest_files}
    bug_priority_map = dict(zip(jira_df['BugID'], jira_df['Priority']))
    print("\n🏷️  开始标注Bug标签（仅时间节点之后的提交）...")

    commit_count = 0
    bug_commit_count = 0
    # 仅遍历时间节点之后的提交
    for commit in repo.iter_commits(after=target_commit):
        commit_count += 1
        msg = commit.message.lower()
        stats = commit.stats.files

        matched_bugs = [bug for bug in bug_priority_map if bug.lower() in msg]
        if not matched_bugs or not stats:
            continue

        bug_commit_count += 1
        # 标注文件标签
        for f in stats.keys():
            f_lower = f.lower()
            if f_lower in latest_files:
                labels[f_lower]["is_buggy"] = 1  # 分类标签：1=Bug文件
                labels[f_lower]["bug_count"] += len(matched_bugs)  # 回归标签：Bug次数

    # 打印标签统计
    buggy_files = sum([1 for f in labels if labels[f]["is_buggy"] == 1])
    total_files = len(labels)
    print(f"遍历节点后提交数: {commit_count}")
    print(f"Bug修复提交数: {bug_commit_count}")
    print(f"标签分布: Bug文件 {buggy_files} 个，非Bug文件 {total_files - buggy_files} 个")
    return labels

# 修改后的函数：仅新增两个标签列，其余完全保留
def save_features_csv(file_stats, bug_stats, labels, output_file=OUTPUT_CSV):
    """保存特征+标签到CSV（仅新增标签列，不修改原有特征）"""
    rows = []
    for f in file_stats:
        # 还原原始文件名（用于展示，不影响特征）
        original_path = None
        for blob in repo.head.commit.tree.traverse():
            if blob.type == "blob" and blob.path.lower() == f:
                original_path = blob.path
                break
        filename = original_path if original_path else f

        row = {
            "filename": filename,
            # 原有特征：完全保留
            "ChangeRate": file_stats[f]["ChangeRate"],
            "ChangeLOC": file_stats[f]["ChangeLOC"],
            "AddLOC": file_stats[f]["AddLOC"],
            "DeleteLOC": file_stats[f]["DeleteLOC"],
            "#Author": file_stats[f]["#Author"],
            "LOC": file_stats[f]["LOC"],
            "BugRate": bug_stats[f]["BugRate"],
            "AvgBugPriority": bug_stats[f]["AvgBugPriority"],
            # 新增标签1：分类标签（是否为Bug文件）
            "is_buggy": labels[f]["is_buggy"],
            # 新增标签2：回归标签（Bug数量）
            "bug_count": labels[f]["bug_count"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 包含标签的训练集已保存到: {output_file}")

    # 打印数据集统计（验证结果）
    print("\n📊 数据集统计:")
    print(f"总样本数: {len(df)}")
    print(f"分类标签分布: {df['is_buggy'].value_counts().to_dict()}（0=非Bug，1=Bug）")
    print(f"回归标签统计: 平均Bug数={df['bug_count'].mean():.2f}，最大Bug数={df['bug_count'].max()}")
    print(f"有变更的文件数: {df[df['ChangeRate'] > 0].shape[0]}")
    print(f"有Bug记录的文件数: {df[df['BugRate'] > 0].shape[0]}")

# 主流程：仅新增标签相关步骤，不修改原有流程
if __name__ == "__main__":
    # 1. 克隆/更新仓库（原有逻辑）
    repo = clone_or_update_repo(REPO_URL, LOCAL_REPO_PATH)

    # 2. 获取最新版本文件列表（原有逻辑）
    latest_files = get_latest_files_all(repo)

    # 3. 提取Git变更特征（原有逻辑）
    file_stats = extract_git_features(repo, latest_files)

    # 4. 加载JIRA Bug数据（原有逻辑）
    jira_df = load_jira_bugs_xml(JIRA_BUG_FILE)

    # 5. 提取Bug特征（原有逻辑）
    bug_stats = extract_bug_features(repo, latest_files, jira_df)

    # 新增步骤1：获取时间节点（早期版本）的commit
    target_commit = get_target_version_commit(repo, TARGET_VERSION)

    # 新增步骤2：标注两个标签
    labels = add_defect_labels(repo, latest_files, jira_df, target_commit)

    # 6. 保存特征+标签（修改后：添加标签列）
    save_features_csv(file_stats, bug_stats, labels)
