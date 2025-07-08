import json
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

try:
    import sqlglot
    from sqlglot.expressions import Table, Column, EQ
except ImportError:
    print("错误: 本功能需要 'sqlglot' 库。")
    print("请通过 'pip install sqlglot' 命令安装。")
    exit(1)

def extract_data_to_json(csv_file_path, output_json_path):
    """
    从CSV文件中提取'id', 'req_url', 和 'db_sql'列的内容，并将其保存为JSON文件。
    JSON的键是'id'列的值。
    """
    results_dict = {}
    try:
        chunk_iter = pd.read_csv(csv_file_path, chunksize=10000, on_bad_lines='skip', low_memory=False)
        required_columns = ['id', 'req_url', 'db_sql']
        for i, chunk in enumerate(chunk_iter):
            if not all(col in chunk.columns for col in required_columns):
                print(f"错误: CSV文件中缺少必需的列 (id, req_url, db_sql)。")
                return
            chunk.dropna(subset=['id'], inplace=True)
            chunk['id'] = chunk['id'].astype(str)
            for index, row in chunk.iterrows():
                results_dict[row['id']] = {
                    'req_url': row.get('req_url'),
                    'db_sql': row.get('db_sql')
                }
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(results_dict, jsonfile, ensure_ascii=False, indent=4)
        print(f"成功从 {csv_file_path} 提取数据到 {output_json_path}")
    except FileNotFoundError:
        print(f"错误: 未找到文件 {csv_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        print(traceback.format_exc())

def analyze_sql_relationships(input_json_path, output_json_path, dialect="oracle"):
    """
    使用 sqlglot 的 AST 全面分析SQL，提取规范化的、去重后的表间字段关联。
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"成功加载输入JSON文件: {input_json_path}")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析JSON文件失败 {input_json_path}")
        return

    table_relationships = defaultdict(lambda: defaultdict(set))

    for key, value in tqdm(data.items(), desc=f"使用 {dialect} 方言解析SQL", total=len(data)):
        sql_query = value.get('db_sql')
        if not sql_query or not isinstance(sql_query, str):
            continue

        try:
            expr = sqlglot.parse_one(sql_query, read=dialect)
            if not expr:
                continue

            alias_map = {}
            for t in expr.find_all(Table):
                real_name = t.this.sql(dialect=dialect).upper()  # 真实表名统一大写
                alias_name = (t.alias_or_name or real_name).upper()  # 别名也大写
                alias_map[alias_name] = real_name

            for eq in expr.find_all(EQ):
                if not (isinstance(eq.left, Column) and isinstance(eq.right, Column)):
                    continue

                l_alias = (eq.left.table or "").upper()
                r_alias = (eq.right.table or "").upper()

                l_table = alias_map.get(l_alias)
                r_table = alias_map.get(r_alias)
                if not (l_table and r_table and l_table != r_table):
                    continue

                # 列名统一大写
                l_col = eq.left.name.upper()
                r_col = eq.right.name.upper()
                op = "="                                   # EQ 节点恒为等号

                # 按字典序交换，确保 A=B 与 B=A 只保留一条
                if (l_table, l_col) > (r_table, r_col):
                    l_table, r_table = r_table, l_table
                    l_col,   r_col   = r_col,   l_col

                canonical = f"{l_table}.{l_col} {op} {r_table}.{r_col}"
                table_relationships[l_table][r_table].add(canonical)

        except Exception as e:
            print(f"\n警告: 解析ID为 '{key}' 的SQL时出错。错误: {e}")

    final_relationships = defaultdict(dict)
    for table1, related_tables in table_relationships.items():
        for table2, conditions in related_tables.items():
            # 过滤掉指向自身的空关系
            if table1 != table2:
                final_relationships[table1][table2] = sorted(list(conditions))

    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(final_relationships, jsonfile, ensure_ascii=False, indent=4)

    print(f"\n成功分析SQL关系并保存到 {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从CSV提取数据或使用 sqlglot 分析SQL表关系。")
    subparsers = parser.add_subparsers(dest='command', help='可执行的命令')

    parser_extract = subparsers.add_parser('extract', help='从CSV提取数据到JSON。')
    parser_extract.add_argument("csv_path", help="输入的CSV文件路径。")
    parser_extract.add_argument("json_path", help="输出的JSON文件路径。")

    parser_analyze = subparsers.add_parser('analyze', help='从JSON文件分析SQL表关系。')
    parser_analyze.add_argument("input_json", help="包含SQL查询的输入JSON文件。")
    parser_analyze.add_argument("output_json", help="用于保存表关系图的输出JSON文件。")
    parser_analyze.add_argument("--dialect", default="mysql", help="要使用的SQL方言 (例如: oracle, mysql, postgres)。默认为 'oracle'。")

    args = parser.parse_args()

    if args.command == 'extract':
        extract_data_to_json(args.csv_path, args.json_path)
    elif args.command == 'analyze':
        analyze_sql_relationships(args.input_json, args.output_json, dialect=args.dialect)
    else:
        parser.print_help()