import csv
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
from DataUtils import EyeMovement, DataUtils
from sklearn.ensemble import RandomForestClassifier
from config.settings import *
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from scipy.stats import pearsonr, chi2_contingency
from sklearn.model_selection import StratifiedKFold

def merge_csv_files(file1, file2, output_file, on='id'):
    """
    合并两个 CSV 文件，合并相同 id 的行
    :param file1: 第一个 CSV 文件路径
    :param file2: 第二个 CSV 文件路径
    :param output_file: 合并后的输出文件路径
    :param on: 合并关键字，默认为 'id'
    """
    # 读取两个 CSV 文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 合并两个 DataFrame，根据指定的关键字
    merged_df = pd.merge(df1, df2, on=on, how='inner')  # 使用 inner join 确保只保留有共同 id 的行

    # 将合并后的结果保存到新的 CSV 文件
    merged_df.to_csv(output_file, index=False)

    print(f"合并后的文件已保存到 {output_file}")

def processMMSE(file_path):
    # 读取 CSV 文件
    data = pd.read_csv(file_path)  # 替换为你的文件路径

    # 定义一个函数来判断是否患病
    def diagnose_mci(row):
        if row['edu'] == 2 and row['MMSE'] >= 24:
            return 0  # 未患病
        elif row['edu'] == 1 and row['MMSE'] >= 20:
            return 0  # 未患病
        elif row['edu'] == 0 and row['MMSE'] >= 17:
            return 0  # 未患病
        else:
            return 1  # 患病

    # 添加新列 '是否患病'，并根据规则赋值
    data['MCI'] = data.apply(diagnose_mci, axis=1)

    # 保存处理后的 CSV 文件
    data.to_csv(file_path, index=False)  # 替换为你想保存的文件路径

def process_all_eye_tracking_files(folder_path, choice=0, maxvel=100):
    """
    处理 eye_data 文件夹中的所有眼动数据文件
    :param folder_path: eye_data 文件夹路径
    :param maxvel: I-VT 算法的速度阈值，默认为 100
    :return: 字典，key 为文件名，value 为 res_dict
    """
    results = {}  # 存储所有文件的结果
    # 记录异常被试者的 id
    error_subjects = []
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):  # 只处理 .txt 文件
            try:
                file_path = os.path.join(folder_path, file_name)  # 文件完整路径
                # 处理文件并获取 res_dict
                res_dict = process_eye_tracking_file(file_path, choice, maxvel)
                # 将文件名（不含扩展名）作为 key，res_dict 作为 value
                print(file_name.split('.')[0])
                print(res_dict)
                results[file_name.split('.')[0]] = res_dict
            except Exception as e:
                # 捕获异常，记录错误被试者 id 和错误信息
                error_subjects.append((file_name.split('.')[0], str(e)))
                print(f"处理被试 {file_name.split('.')[0]} 时发生错误: {e}")
                continue  # 跳过当前被试者，继续处理下一个

    if error_subjects:
        print("\n以下被试者处理失败：")
        for subject_id, error_msg in error_subjects:
            print(f"- 被试 {subject_id}: {error_msg}")
    else:
        print("所有被试者数据处理完成，未发生错误。")

    return results

# 统计和模型特征重要性分析
def feature_importance():
    # 加载数据
    df = pd.read_csv('feature_data/result1.csv')
    X = df.drop(['id', 'MCI'], axis=1)
    y = df['MCI']
    # 区分分类/连续特征类型
    cat_features = ['edu', 'gender']
    cont_features = [col for col in X.columns if col not in cat_features]

    # 标准化连续特征（用于方差筛选和后续建模）
    scaler = RobustScaler()
    X_cont_scaled = scaler.fit_transform(X[cont_features])
    X_scaled = pd.concat([
        pd.DataFrame(X_cont_scaled, columns=cont_features),
        X[cat_features]
    ], axis=1)

    # (1) 互信息筛选（适用于非线性关系）
    # 计算互信息得分（离散特征需标注）
    mi_scores = mutual_info_classif(
        X_scaled, y,
        discrete_features=[X_scaled.columns.get_loc(c) for c in cat_features]
    )
    mi_df = pd.DataFrame({
        'Feature': X_scaled.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)

    print("\n=== 互信息得分 ===")
    print(mi_df.round(4))

    # 可视化
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mi_df, x='MI_Score', y='Feature', palette='viridis')
    plt.title('Mutual Information Scores (Higher=More Important)')
    plt.xlabel('MI Score')
    adjust_params = {
        'left': 0.15,  # 优先调整这个（0.15-0.3）
        'right': 0.95,  # 必要时略微减小右侧
        'top': 0.95,  # 必要时略微减小顶部
        'bottom': 0.05  # 必要时略微增加底部
    }
    plt.tight_layout(rect=(
        adjust_params['left'],
        adjust_params['bottom'],
        adjust_params['right'],
        adjust_params['top']
    ))
    plt.show()


    # 分类特征卡方检验（gender/edu）
    # 对每个分类特征计算卡方统计量和p值

    chi2_results = []
    for col in cat_features:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(X[col], y))
        chi2_results.append({'Feature': col, 'Chi2': chi2, 'P-value': p})
    chi2_df = pd.DataFrame(chi2_results)
    print("\n=== 卡方检验结果 ===")
    print(chi2_df.round(4))
    # 可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    sns.barplot(data=chi2_df, x='Feature', y='Chi2', palette='Blues_d')
    plt.title('Chi2 Statistics (Higher=More Important)')

    plt.subplot(122)
    sns.barplot(data=chi2_df, x='Feature', y='P-value', palette='Reds_d')
    plt.axhline(0.05, color='black', linestyle='--')
    plt.title('P-values (Lower=More Significant)')
    plt.tight_layout()
    plt.show()

    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    # 提取重要性
    rf_importances = pd.DataFrame({
        'Feature': X_scaled.columns,
        'RF_Importance': rf.feature_importances_
    }).sort_values('RF_Importance', ascending=False)

    # 打印原始数值（保留6位小数）
    print("=== 随机森林特征重要性（原始值）===")
    print(rf_importances.round(6).to_string(index=False))

    # 打印标准化后的重要性（0-1范围）
    print("\n=== 归一化后的重要性（总和=1）===")
    print(rf_importances.assign(
        Normalized=rf_importances['RF_Importance'] / rf_importances['RF_Importance'].sum()
    ).round(4).to_string(index=False))

    # 可视化
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rf_importances, x='RF_Importance', y='Feature', palette='rocket')
    plt.title('Random Forest Feature Importances (Higher=More Important)')
    adjust_params = {
        'left': 0.2,  # 优先调整这个（0.15-0.3）
        'right': 0.95,  # 必要时略微减小右侧
        'top': 0.95,  # 必要时略微减小顶部
        'bottom': 0.05  # 必要时略微增加底部
    }
    plt.tight_layout(rect=(
        adjust_params['left'],
        adjust_params['bottom'],
        adjust_params['right'],
        adjust_params['top']
    ))
    plt.show()



    # 计算SHAP值
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # 打印原始SHAP值到终端
    print("=== SHAP值原始数据 ===")
    print(f"SHAP值矩阵形状: {np.array(shap_values).shape}")  # (n_classes, n_samples, n_features)

    # 二分类任务取第一个类别的SHAP值（类别0）
    shap_values_class0 = np.array(shap_values)[:, :, 0]  # 形状变为(1236, 17)

    # 计算平均绝对SHAP值
    shap_abs_mean_values = np.abs(shap_values_class0).mean(axis=0)  # 形状(17,)

    # 创建DataFrame
    shap_abs_mean = pd.DataFrame({
        'Feature': X_scaled.columns,
        'SHAP_abs_mean': shap_abs_mean_values
    }).sort_values('SHAP_abs_mean', ascending=False)

    # 打印结果
    print("\n=== 类别0（MCI=0）的特征SHAP重要性 ===")
    print(shap_abs_mean.round(4))

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=shap_abs_mean,
        x='SHAP_abs_mean',
        y='Feature',
        palette='viridis'
    )
    plt.title('SHAP Feature Importance (for Class 0: Non-MCI)')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.show()


    # 合并所有评分
    importance_df = pd.DataFrame({'Feature': X_scaled.columns})
    print("=== importance_df 预览 ===")
    print(importance_df.head())

    # 归一化到0-1范围
    methods = {
        'MI': mi_df.set_index('Feature')['MI_Score'],
        'RF': rf_importances.set_index('Feature')['RF_Importance'],
        'SHAP': pd.Series(np.abs(shap_values[:, :, 1]).mean(axis=0), index=X_scaled.columns)
    }

    importance_df = pd.DataFrame({'Feature': X_scaled.columns})
    for name, scores in methods.items():
        importance_df[name] = scores.values  # 确保直接赋值数值

    # 验证维度一致性
    for name, scores in methods.items():
        print(f"{name} 长度: {len(scores)}")  # 应该都是17


    # 3. 归一化到[0,1]范围
    for col in methods.keys():
        col_min = importance_df[col].min()
        col_max = importance_df[col].max()
        if col_max != col_min:  # 避免除以0
            importance_df[col] = (importance_df[col] - col_min) / (col_max - col_min)
        else:
            importance_df[col] = 0.5  # 所有值相同则设为中值


    # 计算综合评分（平均归一化得分）
    importance_df['Composite_Score'] = importance_df[list(methods.keys())].mean(axis=1)
    importance_df = importance_df.sort_values('Composite_Score', ascending=False)


    # 打印结果
    print("\n=== 特征重要性评分（不含卡方检验）===")
    print(importance_df.round(3))

    # 热力图（仅显示MI/RF/SHAP）
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        importance_df.set_index('Feature').iloc[:, :-1],
        annot=True,
        cmap='YlGnBu',
        fmt=".2f",
        linewidths=0.5
    )
    adjust_params = {
        'left': 0.2,  # 优先调整这个（0.15-0.3）
        'right': 0.95,  # 必要时略微减小右侧
        'top': 0.95,  # 必要时略微减小顶部
        'bottom': 0.05  # 必要时略微增加底部
    }
    plt.tight_layout(rect=(
        adjust_params['left'],
        adjust_params['bottom'],
        adjust_params['right'],
        adjust_params['top']
    ))
    plt.title('Normalized Feature Importance (MI + RF + SHAP)')
    plt.show()

    # 综合评分条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x='Composite_Score',
        y='Feature',
        palette='coolwarm',
        order=importance_df.sort_values('Composite_Score', ascending=False)['Feature']
    )
    plt.axvline(0.5, color='red', linestyle='--')
    plt.yticks(rotation=0, ha='right')  # 水平对齐到右侧
    # 动态调整布局（逐步尝试这些值）
    adjust_params = {
        'left': 0.2,  # 优先调整这个（0.15-0.3）
        'right': 0.95,  # 必要时略微减小右侧
        'top': 0.95,  # 必要时略微减小顶部
        'bottom': 0.05  # 必要时略微增加底部
    }
    plt.tight_layout(rect=(
    adjust_params['left'],
    adjust_params['bottom'],
    adjust_params['right'],
    adjust_params['top']
    ))
    plt.title('Composite Importance Score (Threshold=0.5)')
    plt.show()

def process_eye_tracking_file(file_path, choice, maxvel=100):
    """
    处理眼动数据文件，调用 I-VT/I-HMM/I-DT 算法并返回结果
    :param file_path: 眼动数据文件的路径
    :param choice: 眼动数据注视点识别算法选择 (0: I-VT, 1: I-HMM, 2: I-DT)
    :param maxvel: I-VT 算法的速度阈值，默认为 100
    :return: 包含眼动特征和回视率的字典
    """
    # 读取文件
    gaze_data = pd.read_csv(file_path, sep=',')

    # 初始化工具类和算法类
    util = DataUtils(gaze_data)
    detector = EyeMovement()

    # 计算眼动特征
    feature_dict = process_data_and_calculate_metrics(
        util, detector, Lvl_state1, choice, maxvel, detector.calculate_eyeMovements_metrics
    )

    # 计算回视率
    feature_dict['regression_rate'] = process_data_and_calculate_metrics(
        util, detector, Lvl_state2, choice, maxvel, detector.calculate_regressions
    )
    # 计算首次访问时长
    tmp1 = process_data_and_calculate_metrics(
        util, detector, Lvl_state3, choice, maxvel, detector.calculate_first_pass_time)
    tmp2 = process_data_and_calculate_metrics(
        util, detector, Lvl_state4, choice, maxvel, detector.calculate_first_pass_time)
    if tmp2 and tmp1:
        feature_dict['first_past_time'] = round((tmp1 + tmp2) / 2, 5)
    elif tmp1:
        feature_dict['first_past_time'] = round(tmp1, 5)
    else:
        feature_dict['first_past_time'] = round(tmp2, 5)
    return feature_dict


def run_algorithm(detector, x, y, time, choice, maxvel):
    """
    根据 choice 调用对应的眼动算法
    :param detector: EyeMovement 实例
    :param x: x 坐标数据
    :param y: y 坐标数据
    :param time: 时间戳数据
    :param choice: 算法选择 (0: I-VT, 1: I-HMM, 2: I-DT)
    :param maxvel: I-VT 算法的速度阈值
    :return: 算法结果
    """
    if choice == 0:
        return detector.eye_movements_detector_IVT(x, y, time, maxvel)
    elif choice == 1:
        return detector.eye_movements_detector_IHMM(x, y, time)
    else:
        return detector.eye_movements_detector_IDT(x, y, time)


def process_data_and_calculate_metrics(util, detector, lvl_state, choice, maxvel, metric_func):
    """
    筛选数据并计算指定指标
    :param util: DataUtils 实例
    :param detector: EyeMovement 实例
    :param lvl_state: 筛选条件
    :param choice: 算法选择
    :param maxvel: I-VT 算法的速度阈值
    :param metric_func: 计算指标的函数
    :return: 计算结果
    """
    x, y, time = util.get_lvl_state(util.prepare_data(), lvl_state)
    result = run_algorithm(detector, x, y, time, choice, maxvel)
    return metric_func(result)


def save_eye_tracking_metrics_to_csv(metrics_dict, output_file):
    """
    将眼动指标字典保存为 CSV 文件
    :param metrics_dict: 包含多个被试眼动指标的字典，格式为 {id: {feature1: value1, feature2: value2, ...}}
    :param output_file: 输出的 CSV 文件路径
    """
    # 获取所有被试的 id
    subject_ids = list(metrics_dict.keys())

    # 获取所有 feature 的名称
    feature_names = list(metrics_dict[subject_ids[0]].keys())

    # 打开 CSV 文件并写入数据
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 写入表头：id + feature 名称
        header = ['id'] + feature_names
        writer.writerow(header)

        # 写入每个被试的数据
        for subject_id in subject_ids:
            row = [subject_id]  # 第一列为 id
            for feature in feature_names:
                row.append(metrics_dict[subject_id][feature])  # 添加每个 feature 的值
            writer.writerow(row)

    print(f"数据已保存到 {output_file}")


def feature_preprocess(train_data: pd.DataFrame, normalize=False):
    data = train_data.copy()

    X = data.drop(columns=['MCI', 'id'])  # 选择除了 'MMSE'/'MCI' 和 'id' 之外的所有列作为特征
    y = data['MCI']  # 选择 'MCI' 列作为标签

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    if normalize:
        # edu_col = 0
        #
        # 分离 edu 列
        # edu_train = X_train[:, 0]  # 训练集的 edu 列
        # edu_test = X_test[:, 0]  # 测试集的 edu 列
        # gen_train = X_train[:, 1]  # 训练集的 edu 列
        # gen_test = X_test[:, 1]  # 测试集的 edu 列

        # columns_to_remove = [0,1]  # 要删除的列的索引
        # X_train = np.delete(X_train, columns_to_remove, axis=1)
        # X_test = np.delete(X_test,columns_to_remove,axis=1)

        scaler = StandardScaler()
        # scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 将 edu 列重新合并到标准化后的数据中
        # X_train = np.column_stack((X_train, edu_train,gen_train))  # 合并训练集
        # X_test = np.column_stack((X_test, edu_test,gen_test))  # 合并测试集 edu 列被添加到最后一列

    return X_train, \
           X_test, \
           y_train.astype(float).flatten(), \
           y_test.astype(float).flatten()


def feature_preprocess_fold(train_data: pd.DataFrame, normalize=False, n_splits=5):
    data = train_data.copy()
    X = data.drop(columns=['MCI', 'id'])  # 选择除了 'MCI' 和 'id' 之外的所有列作为特征
    y = data['MCI']  # 选择 'MCI' 列作为标签

    # 初始化交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储每折的数据
    folds = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test = X_train.values, X_test.values
        y_train, y_test = y_train.values, y_test.values

        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        folds.append((X_train, X_test,
                      y_train.astype(float).flatten(),
                      y_test.astype(float).flatten()))

    return folds