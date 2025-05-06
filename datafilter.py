import pandas as pd
import os


def process_single_file(file_path):
    """处理单个文件，返回各level和state组合的总持续时间"""
    # 读取带表头的文件，明确指定使用前三列
    df = pd.read_csv(
        file_path,
        usecols=['timestamp', 'level', 'state'],  # 直接使用列名选取
        dtype={'timestamp': 'int64', 'level': 'int', 'state': 'int'}
    )

    # 标记连续区间（相同level和state的连续行）
    df['new_interval'] = (df['level'] != df['level'].shift(1)) | (df['state'] != df['state'].shift(1))
    df['interval_id'] = df['new_interval'].cumsum()

    # 计算每个区间的duration
    intervals = df.groupby(['interval_id', 'level', 'state']).agg(
        duration=('timestamp', lambda x: x.iloc[-1] - x.iloc[0])
    ).reset_index()

    # 按level和state汇总总duration
    summary = intervals.groupby(['level', 'state'])['duration'].sum().unstack(fill_value=0)
    return summary

def process_directory_file():
    # 主流程
    input_folder = "eye_data"  # 替换为你的文件夹路径
    output_file = "data/duration.csv"

    # 收集所有结果
    all_results = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            sample_id = os.path.splitext(file_name)[0]  # 用文件名作为id

            try:
                durations = process_single_file(file_path)

                # 动态生成所有可能的level和state组合
                row = {'id': sample_id}
                # 遍历所有level和state组合
                for level in durations.index:  # 动态获取实际存在的level
                    for state in durations.columns:  # 动态获取实际存在的state
                        col_name = f"level{level}_state{state}_dur"
                        row[col_name] = durations.loc[level, state]

                # 填充可能缺失的level/state组合为0
                for level in range(1, 12):  # 包含level11
                    for state in [0, 1, 2]:
                        col_name = f"level{level}_state{state}_dur"
                        if col_name not in row:
                            row[col_name] = 0

                all_results.append(row)

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

    # 保存到CSV（确保列顺序一致）
    output_df = pd.DataFrame(all_results)
    output_df.to_csv(output_file, index=False)
    print(f"处理完成！结果已保存到 {output_file}")




def filter_csv_by_ids(id_file_path, data_file_path, matched_output_path, unmatched_output_path):
    """
    根据ID文件筛选数据文件，将匹配和不匹配的行分别写入两个新文件

    参数:
        id_file_path: 包含ID集合的CSV文件路径
        data_file_path: 包含完整数据的CSV文件路径
        matched_output_path: 匹配行的输出文件路径
        unmatched_output_path: 不匹配行的输出文件路径
    """
    # 读取ID文件
    id_df = pd.read_csv(id_file_path)
    ids = set(id_df['id'])  # 转换为集合提高查找效率

    # 读取数据文件
    data_df = pd.read_csv(data_file_path)

    # 筛选匹配和不匹配的行
    matched_rows = data_df[~data_df['id'].isin(ids)]
    unmatched_rows = data_df[data_df['id'].isin(ids)]

    # 写入输出文件
    matched_rows.to_csv(matched_output_path, index=False)
    unmatched_rows.to_csv(unmatched_output_path, index=False)

    print(f"处理完成。匹配的行数: {len(matched_rows)}，不匹配的行数: {len(unmatched_rows)}")






if __name__=="__main__":

    #计算各样本各阶段持续时长
    # process_directory_file()

    #将异常样本与正常样本分开

    id_file = "data/bizare/bizare_id_134.csv"  # 包含ID集合的文件
    data_file = "feature_data/result3/result3.csv"  # 包含完整数据的文件
    matched_output = "feature_data/result3_filter134.csv"  # 匹配行的输出文件
    unmatched_output = "feature_data/unmatched_97_2.csv"  # 不匹配行的输出文件

    filter_csv_by_ids(id_file, data_file, matched_output, unmatched_output)




