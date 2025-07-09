import csv
import json


def parse_csv_to_json(csv_file_path, json_file_path):
    """
    解析一个特定格式的CSV文件，其中包含表结构信息，并将其转换为JSON。
    输出的JSON是一个字典，键是表名。

    Args:
        csv_file_path (str): 输入的CSV文件路径。
        json_file_path (str): 输出的JSON文件路径。
    """
    tables_dict = {}
    current_table_name = None
    current_table_data = None
    in_fields_section = False

    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not any(field.strip() for field in row):  # 跳过空行
                    if in_fields_section:
                        in_fields_section = False
                        if current_table_name and current_table_data:
                            tables_dict[current_table_name] = current_table_data
                            current_table_name = None
                            current_table_data = None
                    continue

                # 新的表定义
                if row and row[0].strip() == '表名':
                    if current_table_name and current_table_data:
                        tables_dict[current_table_name] = current_table_data

                    table_name = row[1].strip() if len(row) > 1 else ''
                    if table_name:
                        current_table_name = table_name
                        current_table_data = {
                            'description': '',
                            'fields': []
                        }
                    in_fields_section = False
                    continue

                if current_table_data:
                    # 表描述
                    if row and row[0].strip() == '描述':
                        current_table_data['description'] = row[1].strip() if len(row) > 1 else ''
                        continue

                    # 字段表头
                    if row and row[0].strip() == '字段明细':
                        in_fields_section = True
                        # 固定的表头，可以直接跳过
                        continue

                    # 字段数据
                    if in_fields_section and len(row) > 5 and row[1].strip():
                        field_info = {
                            'field_name': row[1].strip(),
                            'field_type': row[2].strip(),
                            'is_nullable': row[3].strip(),
                            'default_value': row[4].strip(),
                            'field_description': row[5].strip()
                        }
                        current_table_data['fields'].append(field_info)

        # 添加最后一个表
        if current_table_name and current_table_data:
            tables_dict[current_table_name] = current_table_data

        # 写入JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(tables_dict, jsonfile, ensure_ascii=False, indent=4)
            
        print(f"成功将 {csv_file_path} 转换为 {json_file_path}")

    except FileNotFoundError:
        print(f"错误: 未找到文件 {csv_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


if __name__ == '__main__':
    # 使用绝对路径以确保脚本在任何位置都能正确运行
    csv_path = '/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/表结构.csv'
    json_path = 'table_info.json'
    parse_csv_to_json(csv_path, json_path)
