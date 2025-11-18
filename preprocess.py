import os
import git
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

repo_url = "https://github.com/apache/avro.git"
repo_path = "./avro_repo"
JIRA_BUG_FILE = "data/AVRO_Bugs_merged.xml"
OUTPUT_CSV = "defect_prediction_dataset.csv"
TARGET_VERSION = "release-1.9.2"


def clone_or_update_repo(repo_url, local_path):
    if not os.path.exists(local_path):
        print("Cloning repository")
        repo = git.Repo.clone_from(
            repo_url, local_path,
            multi_options=["--recursive"]
        )
        repo.git.fetch("--tags")
    else:
        repo = git.Repo(local_path)
    return repo


def get_target_version_commit(repo, target_version):
    tags = {tag.name: tag for tag in repo.tags}
    if target_version not in tags:
        tags_lower = {name.lower(): tag for name, tag in tags.items()}
        target_lower = target_version.lower()
        target_tag = tags_lower[target_lower]
    else:
        target_tag = tags[target_version]

    target_commit = target_tag.commit
    return target_commit



def extract_git_features(repo, target_files, target_commit):
    file_stats = defaultdict(lambda: {
        "ChangeRate": 0,
        "ChangeLOC": 0,
        "AddLOC": 0,
        "DeleteLOC": 0,
        "#Author": set(),
        "LOC": 0
    })

    commit_count = 0
    matched_file_count = 0

    print("\nExtracting Git features")
    total_commits = sum(1 for _ in repo.iter_commits(until=target_commit, reverse=True))
    for commit in tqdm(repo.iter_commits(until=target_commit, reverse=True), total=total_commits, desc="处理提交"):
        commit_count += 1
        author = commit.author.email
        stats = commit.stats.files

        if not stats:
            continue

        for f, s in stats.items():
            f_lower = f.lower()
            if f_lower in target_files:
                matched_file_count += 1
                file_stats[f_lower]["ChangeRate"] += 1
                file_stats[f_lower]["ChangeLOC"] += s['lines']
                file_stats[f_lower]["AddLOC"] += s['insertions']
                file_stats[f_lower]["DeleteLOC"] += s['deletions']
                file_stats[f_lower]["#Author"].add(author)

    print(f"目标版本前处理的提交数: {commit_count}")
    print(f"目标版本前匹配的文件变更数: {matched_file_count}")
    print(f"有Git特征的文件数: {len([f for f in file_stats if file_stats[f]['ChangeRate'] > 0])}")

    print("Calculating LOC for target version files...")
    for f_lower in tqdm(target_files, desc="计算文件LOC"):
        original_path = None
        for blob in target_commit.tree.traverse():
            if blob.type == "blob" and blob.path.lower() == f_lower:
                original_path = blob.path
                break

        if original_path:
            try:
                blob = target_commit.tree[original_path]
                blob_data = blob.data_stream.read()
                lines = blob_data.decode('utf-8', errors='ignore').splitlines()
                file_stats[f_lower]["LOC"] = len(lines)
            except Exception as e:
                print(f"Failed to read {original_path}: {e}")
                file_stats[f_lower]["LOC"] = 0
        else:
            file_stats[f_lower]["LOC"] = 0

    for f in file_stats:
        file_stats[f]["#Author"] = len(file_stats[f]["#Author"])

    for f_lower in target_files:
        if f_lower not in file_stats:
            file_stats[f_lower] = {
                "ChangeRate": 0,
                "ChangeLOC": 0,
                "AddLOC": 0,
                "DeleteLOC": 0,
                "#Author": 0,
                "LOC": 0
            }

    return file_stats


def load_jira_bugs_xml(jira_file):
    tree = ET.parse(jira_file)
    root = tree.getroot()
    priority_map = {"Blocker": 5, "Critical": 4, "Major": 3, "Minor": 2, "Trivial": 1}
    bug_list = []

    items = root.findall(".//item")
    for item in tqdm(items, desc="加载Bug数据"):
        bug_id = item.findtext("key")
        bug_type = item.findtext("type")
        priority_text = item.findtext("priority")

        if bug_id and bug_type and priority_text:
            priority_text = priority_text.strip()
            priority_value = priority_map.get(priority_text, 1)
            bug_list.append({
                "BugID": bug_id.strip(),
                "Type": bug_type.strip(),
                "Priority": priority_value
            })

    df = pd.DataFrame(bug_list)
    print(f"\n加载到 {len(df)} 个Bug记录")
    return df


def extract_bug_features(repo, target_files, jira_df, target_commit):
    bug_stats = {f: {"BugRate": 0, "AvgBugPriority": 0} for f in target_files}
    bug_priority_map = dict(zip(jira_df['BugID'], jira_df['Priority']))
    commit_count = 0
    matched_bug_commit_count = 0

    print("\nExtracting Bug features")
    total_commits = sum(1 for _ in repo.iter_commits(until=target_commit, reverse=True))
    for commit in tqdm(repo.iter_commits(until=target_commit, reverse=True), total=total_commits, desc="处理提交"):
        commit_count += 1
        msg = commit.message.lower()

        matched_bugs = [bug for bug in bug_priority_map if bug.lower() in msg]
        if not matched_bugs:
            continue

        matched_bug_commit_count += 1
        stats = commit.stats.files

        if not stats:
            continue

        for f, _ in stats.items():
            f_lower = f.lower()
            if f_lower in target_files:
                for bug in matched_bugs:
                    bug_stats[f_lower]["BugRate"] += 1
                    bug_stats[f_lower]["AvgBugPriority"] += bug_priority_map[bug]

    for f in bug_stats:
        if bug_stats[f]["BugRate"] > 0:
            bug_stats[f]["AvgBugPriority"] = round(bug_stats[f]["AvgBugPriority"] / bug_stats[f]["BugRate"], 2)
        else:
            bug_stats[f]["AvgBugPriority"] = 0

    print(f"目标版本前处理的提交数: {commit_count}")
    print(f"目标版本前包含BugID的提交数: {matched_bug_commit_count}")
    return bug_stats


def add_defect_labels(repo, target_files, jira_df, target_commit):
    labels = {f: {"is_buggy": 0, "bug_count": 0} for f in target_files}
    bug_priority_map = dict(zip(jira_df['BugID'], jira_df['Priority']))

    commit_count = 0
    bug_commit_count = 0

    print("\nLabeling buggy files")
    total_commits = sum(1 for _ in repo.iter_commits(after=target_commit, reverse=True))
    for commit in tqdm(repo.iter_commits(after=target_commit, reverse=True), total=total_commits, desc="处理提交"):
        commit_count += 1
        msg = commit.message.lower()
        stats = commit.stats.files

        matched_bugs = [bug for bug in bug_priority_map if bug.lower() in msg]
        if not matched_bugs or not stats:
            continue

        bug_commit_count += 1
        for f in stats.keys():
            f_lower = f.lower()
            if f_lower in target_files:
                labels[f_lower]["is_buggy"] = 1
                labels[f_lower]["bug_count"] += len(matched_bugs)

    buggy_files = sum([1 for f in labels if labels[f]["is_buggy"] == 1])
    total_files = len(labels)
    print(f"目标版本后遍历的提交数: {commit_count}")
    print(f"目标版本后Bug修复提交数: {bug_commit_count}")
    print(f"标签分布: Bug文件 {buggy_files} 个，非Bug文件 {total_files - buggy_files} 个")
    return labels


def save_features_csv(file_stats, bug_stats, labels, target_commit, output_file=OUTPUT_CSV):
    rows = []
    target_version = TARGET_VERSION

    for f in tqdm(file_stats, desc="生成CSV数据"):
        original_path = None
        for blob in target_commit.tree.traverse():
            if blob.type == "blob" and blob.path.lower() == f:
                original_path = blob.path
                break
        filename = original_path if original_path else f

        row = {
            "filename": filename,
            "ChangeRate": file_stats[f]["ChangeRate"],
            "ChangeLOC": file_stats[f]["ChangeLOC"],
            "AddLOC": file_stats[f]["AddLOC"],
            "DeleteLOC": file_stats[f]["DeleteLOC"],
            "#Author": file_stats[f]["#Author"],
            "LOC": file_stats[f]["LOC"],
            "BugRate": bug_stats[f]["BugRate"],
            "AvgBugPriority": bug_stats[f]["AvgBugPriority"],
            "is_buggy": labels[f]["is_buggy"],
            "bug_count": labels[f]["bug_count"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("数据集统计:")
    print(f"目标版本: {target_version}")
    print(f"总样本数: {len(df)}")
    print(f"分类标签分布: {df['is_buggy'].value_counts().to_dict()}（0=非Bug，1=Bug）")
    print(f"回归标签统计: 平均Bug数={df['bug_count'].mean():.2f}，最大Bug数={df['bug_count'].max()}")
    print(f"数据集已保存到: {output_file}")


if __name__ == "__main__":
    repo = clone_or_update_repo(repo_url, repo_path)
    target_commit = get_target_version_commit(repo, TARGET_VERSION)
    target_files = get_target_version_files(repo, target_commit)
    file_stats = extract_git_features(repo, target_files, target_commit)
    jira_df = load_jira_bugs_xml(JIRA_BUG_FILE)
    bug_stats = extract_bug_features(repo, target_files, jira_df, target_commit)
    labels = add_defect_labels(repo, target_files, jira_df, target_commit)
    save_features_csv(file_stats, bug_stats, labels, target_commit)
