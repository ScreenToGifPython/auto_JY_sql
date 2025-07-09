# -*- encoding: utf-8 -*-
"""
@File: merge_table_json.py
@Modify Time: 2025/7/9 08:07
@Author: Kevin-Chen
@Descriptions: Merge table information and relation files, and govern the relation data to ensure symmetry and correct order.
"""
import json
from collections import defaultdict


def merge_and_govern_relations(info_path, relation_path, output_path):
    """
    Merges table information and relations from two JSON files,
    then governs the relations to ensure data integrity and consistency.
    """
    # Step 1: Read the two JSON files
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            table_info = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Information file not found at {info_path}")
        return

    try:
        with open(relation_path, 'r', encoding='utf-8') as f:
            table_relations = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Relation file not found at {relation_path}")
        return

    # Step 2: Perform the initial merge
    merged_data = {}
    for table_name, info in table_info.items():
        merged_data[table_name] = {
            "description": info.get("description", ""),
            "fields": info.get("fields", []),
            "relations": table_relations.get(table_name, {})
        }

    for table_name, rel in table_relations.items():
        if table_name not in merged_data:
            merged_data[table_name] = {
                "description": "",
                "fields": [],
                "relations": rel
            }

    # Step 3: Govern the relations data
    # Collect all unique relations to avoid duplicates and handle asymmetry
    canonical_relations = set()
    for data in merged_data.values():
        for conditions in data.get("relations", {}).values():
            for condition in conditions:
                parts = [p.strip() for p in condition.split('=')]
                if len(parts) == 2:
                    # Store as a sorted tuple to treat 'A=B' and 'B=A' as the same
                    canonical_relations.add(tuple(sorted(parts)))

    # Rebuild relations symmetrically and in the correct order
    governed_relations = defaultdict(lambda: defaultdict(list))
    for rel_tuple in canonical_relations:
        part1, part2 = rel_tuple
        table1_name = part1.split('.')[0]
        table2_name = part2.split('.')[0]

        # Ensure both tables involved in the relation exist in the merged data
        if table1_name in merged_data and table2_name in merged_data:
            # Add relation for table1: table1.col = table2.col
            governed_relations[table1_name][table2_name].append(f"{part1} = {part2}")
            # Add symmetric relation for table2: table2.col = table1.col
            governed_relations[table2_name][table1_name].append(f"{part2} = {part1}")

    # Update merged_data with the new, governed relations
    for table_name, relations in governed_relations.items():
        if table_name in merged_data:
            merged_data[table_name]["relations"] = dict(relations)

    # Step 4: Save the final, governed data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 合并与治理完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    # Define absolute paths for the files
    base_path = "/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer"
    info_file = f"{base_path}/table_info.json"
    relation_file = f"{base_path}/table_relation.json"
    output_file = f"{base_path}/table.json"

    merge_and_govern_relations(info_file, relation_file, output_file)
