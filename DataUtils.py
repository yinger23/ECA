import numpy as np
import pandas as pd
from math import pi, log, sqrt, degrees, atan2
from hmmlearn import hmm
from config.settings import *


class DataUtils():
    def __init__(self, s):
        self._s = s

    def prepare_data(self) -> pd.DataFrame:
        df = self._s.copy()
        df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
        df['level'] = df['level'].astype(int, errors='ignore')
        df['state'] = df['state'].astype(int, errors='ignore')
        df['pos_x'] = df['pos_x'].astype(float, errors='ignore')
        df['pos_y'] = df['pos_y'].astype(float, errors='ignore')
        df['left'] = df['left'].astype(float, errors='ignore')
        df['right'] = df['right'].astype(float, errors='ignore')
        start_time = df['timestamp'].min()
        df['timestamp'] = df['timestamp'] - start_time
        na_idx = df.isna().sum(axis=1) == 0
        df = df[na_idx]
        return df

    def get_lvl_state(self, df: pd.DataFrame, lvl_state_pairs: list):
        """
        从df中筛选指定的level和state组合的数据，并合并结果。
        参数:
        - df: 输入的DataFrame。
        - lvl_state_pairs: 包含level和state组合的列表，格式为[[level1, state1], [level2, state2], ...]。
        返回:
        - x, y, time: 合并后的pos_x, pos_y, timestamp数组。
        """
        # 初始化空列表存储结果
        x_list, y_list, time_list = [], [], []

        # 遍历每个level和state组合
        for level, state in lvl_state_pairs:
            idx1 = df['level'] == level
            idx2 = df['state'] == state
            idx = idx1 & idx2
            tmpDf = df[idx].copy()

            # 将筛选结果添加到列表中
            x_list.extend(tmpDf['pos_x'].values)
            y_list.extend(tmpDf['pos_y'].values)
            time_list.extend(tmpDf['timestamp'].values)

        # 将列表转换为numpy数组
        x = np.array(x_list)
        y = np.array(y_list)
        time = np.array(time_list)

        return x, y, time


class EyeMovement():
    def __init__(self):
        pass

    # 基于速度做注视点识别中速度计算
    def calculate_velocity(self, x, y, time):
        # np.diff 差分函数 做数组中元素减法（后减前）
        x_diff = np.diff(x)
        y_diff = np.diff(y)
        time_int = np.diff(time)
        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
        ang_diff = np.arctan(dist / 5.9) * 180 / np.pi  # 这里 5.9 可能是屏幕到眼睛的距离（单位：厘米），用于将距离转换为视角
        # np.divide 矩阵（向量）除法
        vel = np.divide(ang_diff, time_int)  # degree / ms   # 速度 = 距离 / 时间
        return vel

    def eye_movements_detector_IVT(self, x, y, time, maxvel=100):
        """
        使用 I-VT 算法将眼动数据分类为注视点和扫视点
        :param x: x坐标序列 numpy array
        :param y: y坐标序列 numpy array
        :param time: 时间戳序列 numpy array
        :param maxvel: 预先定义的速度阈值区分注视点（fixation）和扫视点（saccade）
        :return: dict of event list,
                e.g. the value of 'sac' is a list of lists which contains
                start[0] & end[1] time of each sacade,
                duration[2],
                start[3] & end[4] coordinates of saccade point
        """
        fix_list = []
        sac_list = []
        res_dict = {'fix': [], 'sac': [], 'gap': []}

        vel = self.calculate_velocity(x, y, time)

        threshold_vel = vel.copy()
        for i in range(len(threshold_vel)):
            if threshold_vel[i] < maxvel / 1000:  # 注视点
                if not fix_list and sac_list:  # 结束扫视点序列
                    sac_sx, sac_sy = x[i - len(sac_list)], y[i - len(sac_list)]
                    sac_ex, sac_ey = x[i - 1], y[i - 1]
                    res_dict['sac'].append(
                        [sac_list[0], sac_list[-1], sac_list[-1] - sac_list[0], (sac_sx, sac_sy), (sac_ex, sac_ey)])
                    sac_list = []
                fix_list.append(time[i])
            else:  # 扫视点
                if not sac_list and fix_list:  # 结束注视点序列
                    fix_x = round(np.mean(x[i - len(fix_list):i]), 4)
                    fix_y = round(np.mean(y[i - len(fix_list):i]), 4)
                    res_dict['fix'].append([fix_list[0], fix_list[-1], fix_list[-1] - fix_list[0], (fix_x, fix_y)])
                    fix_list = []
                sac_list.append(time[i])

        # 处理最后一个序列
        if fix_list:
            fix_x = round(np.mean(x[-len(fix_list):]), 4)
            fix_y = round(np.mean(y[-len(fix_list):]), 4)
            res_dict['fix'].append([fix_list[0], fix_list[-1], fix_list[-1] - fix_list[0], (fix_x, fix_y)])
        if sac_list:
            sac_sx, sac_sy = x[-len(sac_list)], y[-len(sac_list)]
            sac_ex, sac_ey = x[-1], y[-1]
            res_dict['sac'].append(
                [sac_list[0], sac_list[-1], sac_list[-1] - sac_list[0], (sac_sx, sac_sy), (sac_ex, sac_ey)])

        return res_dict

    def eye_movements_detector_IHMM(self, x, y, time):
        """
        使用 I-HMM 算法将眼动数据分类为注视点和扫视点
        :param x: x坐标序列 numpy array
        :param y: y坐标序列 numpy array
        :param time: 时间戳序列 numpy array
        :param maxvel: 预先定义的速度阈值（可选）
        :return: dict of event list,
                e.g. the value of 'sac' is a list of lists which contains
                start[0] & end[1] time of each saccade,
                duration[2],
                start[3] & end[4] coordinates of saccade point
        """
        vel = self.calculate_velocity(x, y, time)
        # 准备观测序列（速度和位置）     ！！！ 观测变量只选速度，还是速度+坐标？
        observations = np.column_stack((vel, x[:-1], y[:-1]))  # 速度和位置作为观测值

        # 定义隐马尔可夫模型
        model = hmm.GaussianHMM(
            n_components=2,  # 两个状态：注视点和扫视点
            covariance_type="full",  # 使用完整的协方差矩阵   ！！！观测变量间是否关联？
            n_iter=100  # 最大迭代次数
        )

        # 初始化模型参数          ！！！是否先行设置？
        model.startprob_ = np.array([0.8, 0.2])  # 初始概率
        model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])  # 转移概率
        model.means_ = np.array([[10.0, 1.0, 1.0], [100.0, 10.0, 10.0]])  # 观测均值
        model.covars_ = np.array([
            [[10.0, 1.0, 1.0], [1.0, 1.0, 0.5], [1.0, 0.5, 1.0]],  # 状态 0 的协方差矩阵
            [[100.0, 10.0, 10.0], [10.0, 10.0, 5.0], [10.0, 5.0, 10.0]]  # 状态 1 的协方差矩阵
        ])

        # 训练模型
        model.fit(observations)

        # 预测状态序列
        states = model.predict(observations)

        # 将状态序列转换为眼动事件
        res_dict = {'fix': [], 'sac': [], 'gap': []}
        current_state = states[0]
        start_time = time[0]
        start_index = 0

        for i in range(1, len(states)):
            if states[i] != current_state:
                # 状态切换，记录当前事件
                end_time = time[i]
                end_index = i
                if current_state == 0:  # 注视点
                    fix_x = round(np.mean(x[start_index:end_index]), 4)
                    fix_y = round(np.mean(y[start_index:end_index]), 4)
                    res_dict['fix'].append([start_time, end_time, end_time - start_time, (fix_x, fix_y)])
                else:  # 扫视点
                    sac_sx, sac_sy = x[start_index], y[start_index]
                    sac_ex, sac_ey = x[end_index - 1], y[end_index - 1]
                    res_dict['sac'].append(
                        [start_time, end_time, end_time - start_time, (sac_sx, sac_sy), (sac_ex, sac_ey)])
                # 更新状态
                current_state = states[i]
                start_time = time[i]
                start_index = i

        # 处理最后一个事件
        if current_state == 0:  # 注视点
            fix_x = round(np.mean(x[start_index:]), 4)
            fix_y = round(np.mean(y[start_index:]), 4)
            res_dict['fix'].append([start_time, time[-1], time[-1] - start_time, (fix_x, fix_y)])
        else:  # 扫视点
            sac_sx, sac_sy = x[start_index], y[start_index]
            sac_ex, sac_ey = x[-1], y[-1]
            res_dict['sac'].append([start_time, time[-1], time[-1] - start_time, (sac_sx, sac_sy), (sac_ex, sac_ey)])

        return res_dict

    # 合并相近注视点
    def merge_fixation(self, detected_res: dict, min_dur=60, max_ang_diff=0.5, max_time_interval=75):
        """
        :param detected_res:
        :param min_dur:
        :param max_ang_diff:
        :param max_time_interval:
        :return: fix_num before[0] and after[1] merging, combined fixation list[2]
        """
        fix_list = detected_res['fix']
        prev_num = len(fix_list)
        prev_fix = fix_list[0]
        new_fix_list = [prev_fix]

        for fix in fix_list[1:]:
            # fix[-1]注视点平均坐标 (x,y)
            x, y = fix[-1]
            prev_x, prev_y = prev_fix[-1]
            dist = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
            ang_diff = np.arctan(dist / 5.9) * 180 / pi
            # fix[0] 注视点开始时间 fix[1]注视点结束时间
            time_interval = fix[0] - prev_fix[1]
            if ang_diff <= max_ang_diff and time_interval <= max_time_interval:
                fix_x, fix_y = round((x + prev_x) / 2, 4), round((y + prev_y) / 2, 4)  # other approaches
                # 怎么保证两个注视点之间没有扫视点？
                new_fix = [prev_fix[0], fix[1], fix[1] - prev_fix[0], (fix_x, fix_y)]
                new_fix_list.pop()
                new_fix_list.append(new_fix)
                prev_fix = new_fix
            else:
                new_fix_list.append(fix)
                prev_fix = fix
        # 过滤短注视点
        res_list = [fix for fix in new_fix_list if fix[2] >= min_dur]
        # return prev_num, len(res_list), res_list
        detected_res['fix'] = res_list
        return detected_res

    # 基于分散度做注视点识别
    def eye_movements_detector_IDT(self, x, y, time, dispersion_threshold=50, duration_threshold=100):

        """
        使用 I-DT 算法将眼动数据分类为注视点和扫视点
        :param x: x坐标序列 numpy array
        :param y: y坐标序列 numpy array
        :param time: 时间戳序列 numpy array
        :param dispersion_threshold: 分散度阈值（默认 50）
        :param duration_threshold: 持续时间阈值（默认 100 毫秒）
        :return: dict of event list,
                 e.g. the value of 'fix' is a list of lists which contains
                 start[0] & end[1] time of each fixation,
                 duration[2],
                 centroid[3] coordinates of fixation point
        """
        fixations = []  # 存储检测到的注视点
        saccades = []  # 存储检测到的扫视点
        n = len(x)  # 数据点总数
        i = 0  # 当前窗口起始索引

        while i < n:
            # 初始化窗口，覆盖满足 duration_threshold 的最小点数
            window_start = i
            window_end = i + 1

            # 找到满足 duration_threshold 的最小窗口
            while window_end < n and (time[window_end] - time[window_start]) < duration_threshold:
                window_end += 1

            # 如果窗口结束索引超出范围，退出循环
            if window_end >= n:
                break

            # 计算窗口内点的分散度
            window_x = x[window_start:window_end]
            window_y = y[window_start:window_end]
            dispersion = (np.max(window_x) - np.min(window_x)) + (np.max(window_y) - np.min(window_y))

            # 如果分散度低于阈值，扩展窗口
            if dispersion <= dispersion_threshold:
                while window_end < n and dispersion <= dispersion_threshold:
                    window_end += 1
                    if window_end >= n:
                        break
                    window_x = x[window_start:window_end]
                    window_y = y[window_start:window_end]
                    dispersion = (np.max(window_x) - np.min(window_x)) + (np.max(window_y) - np.min(window_y))

                # 记录注视点
                centroid_x = round(np.mean(window_x), 4)
                centroid_y = round(np.mean(window_y), 4)
                onset_time = time[window_start]
                duration = time[window_end - 1] - time[window_start]

                fixations.append([onset_time, time[window_end - 1], duration, (centroid_x, centroid_y)])

                # 如果当前注视点之前有点未被处理，则记录为扫视点
                if window_start > i:
                    sac_start_time = time[i]
                    sac_end_time = time[window_start]
                    sac_duration = sac_end_time - sac_start_time
                    sac_start_x, sac_start_y = x[i], y[i]
                    sac_end_x, sac_end_y = x[window_start], y[window_start]
                    saccades.append([sac_start_time, sac_end_time, sac_duration, (sac_start_x, sac_start_y),
                                     (sac_end_x, sac_end_y)])

                # 移动窗口起始索引到窗口结束位置
                i = window_end
            else:
                # 如果分散度高于阈值，窗口右移一个点
                i += 1

        # 处理最后一个事件
        if i < n:
            # 如果最后剩余的点未被处理，则记录为扫视点
            sac_start_time = time[i]
            sac_end_time = time[-1]
            sac_duration = sac_end_time - sac_start_time
            sac_start_x, sac_start_y = x[i], y[i]
            sac_end_x, sac_end_y = x[-1], y[-1]
            saccades.append(
                [sac_start_time, sac_end_time, sac_duration, (sac_start_x, sac_start_y), (sac_end_x, sac_end_y)])

        # 返回结果
        res_dict = {'fix': fixations, 'sac': saccades, 'gap': []}  # gap 暂时为空
        return res_dict

    def calculate_fixation_entropy(self, res_dict, image_width, image_height, grid_size=10):
        """
        计算注视点空间分布熵
        :param res_dict: 包含注视点信息的字典
        :param image_width: 图像的宽度
        :param image_height: 图像的高度
        :param grid_size: 将图像划分为 grid_size x grid_size 的网格（默认 10x10）
        :return: 注视点空间分布熵
        """
        # 获取注视点列表
        fixations = res_dict.get('fix', [])
        if not fixations:
            return 0  # 如果没有注视点，熵为 0

        # 初始化网格计数器
        grid_counts = np.zeros((grid_size, grid_size))

        # 计算每个注视点所在的网格
        for fix in fixations:
            fix_x, fix_y = fix[3]  # 注视点的坐标 (fix_x, fix_y)

            # 限制 fix_x 和 fix_y 的范围
            fix_x = max(0, min(fix_x, image_width - 1))  # 确保 fix_x 在 [0, image_width - 1] 范围内
            fix_y = max(0, min(fix_y, image_height - 1))  # 确保 fix_y 在 [0, image_height - 1] 范围内

            # 计算注视点所在的网格索引
            grid_x = min(int(fix_x / image_width * grid_size), grid_size - 1)
            grid_y = min(int(fix_y / image_height * grid_size), grid_size - 1)
            grid_counts[grid_y, grid_x] += 1  # 注意：y 是行索引，x 是列索引

        # 计算每个网格的概率
        total_fixations = len(fixations)
        probabilities = grid_counts / total_fixations

        # 计算熵
        entropy = 0
        for row in probabilities:
            for p in row:
                if p > 0:  # 忽略概率为 0 的网格
                    entropy -= p * log(p, 2)  # 以 2 为底的对数

        return entropy

    # 计算分区域回视率
    def calculate_regressions(self, res_dict):
        """
        计算回视次数
        :param res_dict: 结果字典，包含注视点信息
        :param cur_regions: 区域信息列表，每个区域为 (x_min, y_min, x_max, y_max)
        :return: 回视次数
        """
        fixations = res_dict['fix']  # 获取注视点数据
        regression_count = 0  # 回视次数
        prev_region = None  # 上一个注视点所在的区域
        visited_regions = set()  # 记录已经被注视过的区域

        for fixation in fixations:
            x, y = fixation[3]  # 注视点的坐标 (x, y)
            current_region = None

            # 判断当前注视点属于哪个区域
            for i, region in enumerate(regions):
                x_min, y_min, x_max, y_max = region
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    current_region = i
                    break

            if current_region is not None:
                # 如果当前区域与上一个区域不同，则记录为跨越区域
                if prev_region is not None and current_region != prev_region:
                    # 如果当前区域已经被注视过，则增加回视次数
                    if current_region in visited_regions:
                        regression_count += 1
                    # 更新已访问区域
                    visited_regions.add(current_region)

                # 更新上一个区域
                prev_region = current_region

        return round(regression_count / len(fixations), 5)

    def calculate_first_pass_time(self,res_dict):
        """
        计算每个兴趣区的首次访问时长（连续注视点的总持续时间），并返回平均首次访问时长
        :param res_dict: 眼动数据分类结果，包含注视点和扫视点
        :param regions: 兴趣区列表，每个兴趣区是一个元组 (x_min, y_min, x_max, y_max)
        :return: 平均首次访问时长, 各兴趣区的首次访问时长
        """
        # 初始化每个兴趣区的首次访问时长和是否被访问的标志
        first_pass_times = {i: 0 for i in range(len(regions))}  # 存储每个兴趣区的首次访问时长
        visited_regions = {i: False for i in range(len(regions))}  # 记录每个兴趣区是否被访问过

        # 遍历所有注视点
        fixations = res_dict['fix']
        i = 0
        while i < len(fixations):
            start_time, end_time, duration, (fix_x, fix_y) = fixations[i]

            # 判断当前注视点是否在某个兴趣区内
            current_region = None
            for region_idx, region in enumerate(regions):
                x_min, y_min, x_max, y_max = region
                if x_min <= fix_x <= x_max and y_min <= fix_y <= y_max:
                    current_region = region_idx
                    break

            if current_region is not None:
                # 如果当前兴趣区未被访问过，则开始计算首次访问时长
                if not visited_regions[current_region]:
                    total_duration = 0
                    # 累加当前兴趣区的连续注视点持续时间
                    while i < len(fixations):
                        start_time, end_time, duration, (fix_x, fix_y) = fixations[i]
                        # 检查当前注视点是否仍在当前兴趣区内
                        if (x_min <= fix_x <= x_max and y_min <= fix_y <= y_max):
                            total_duration += duration
                            i += 1
                        else:
                            break
                    # 记录首次访问时长
                    first_pass_times[current_region] = total_duration
                    visited_regions[current_region] = True  # 标记为已访问
                else:
                    i += 1  # 如果兴趣区已被访问过，跳过当前注视点
            else:
                i += 1  # 如果注视点不在任何兴趣区内，跳过当前注视点

        # 计算平均首次访问时长（只计算被访问过的兴趣区）
        valid_first_pass_times = [time for time in first_pass_times.values() if time > 0]
        if valid_first_pass_times:
            average_first_pass_time = sum(valid_first_pass_times) / len(valid_first_pass_times)
        else:
            average_first_pass_time = 0

        return round(average_first_pass_time,5)

    # 计算信息识别/记忆阶段的眼动特征
    def calculate_eyeMovements_metrics(self, res_dict):
        """
        计算注视和扫视相关的眼动特征
        :param res_dict: 包含注视点和扫视点信息的字典
        :return: feature_dict 包含各眼动特征
        """
        feature_dict = {}
        # 获取注视点列表
        fixations = res_dict.get('fix', [])

        # 计算注视次数
        fixation_count = len(fixations)
        feature_dict['fixation_count'] = fixation_count

        # 计算总注视时间
        total_fixation_duration = sum(fix[2] for fix in fixations)  # fix[2] 是持续时间
        feature_dict['total_fixation_duration'] = round(total_fixation_duration, 5)

        # 计算平均注视时间
        if fixation_count > 0:
            average_fixation_duration = total_fixation_duration / fixation_count
        else:
            average_fixation_duration = 0  # 如果没有注视点，平均注视时间为 0
        feature_dict['average_fixation_duration'] = round(average_fixation_duration, 5)

        # 计算注视点空间分布熵
        # 图像的宽度和高度
        fixation_entropy = round(self.calculate_fixation_entropy(res_dict, 6.8, 3.7), 5)
        feature_dict['fixation_entropy'] = fixation_entropy

        # 获取扫视点列表
        saccades = res_dict.get('sac', [])
        if not saccades:
            return feature_dict

        # 初始化变量
        total_saccade_amplitude = 0  # 总扫视幅度
        total_saccade_length = 0  # 总扫视路径长度
        total_saccade_duration = 0  # 总扫视时间
        saccade_angles = []  # 扫视角度列表
        forward_saccade_count = 0  # 前进扫视次数
        forward_saccade_len = 0  # 前向扫视距离

        # 遍历每个扫视点

        for sac in saccades:
            # 提取扫视点的起始和结束坐标
            start_x, start_y = sac[3]
            end_x, end_y = sac[4]

            # 计算扫视幅度（欧氏距离）
            amplitude = sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            total_saccade_amplitude += amplitude

            # 计算扫视路径长度（累加扫视幅度）
            total_saccade_length += amplitude

            # 计算扫视时间
            duration = sac[2]
            total_saccade_duration += duration

            # 计算扫视角度（相对于水平轴）
            delta_x = end_x - start_x
            delta_y = end_y - start_y
            angle = degrees(atan2(delta_y, delta_x))  # 计算角度（-180° 到 180°）
            saccade_angles.append(angle)

            # 判断是否为前进扫视（x 坐标增加）
            if delta_x > 0:
                forward_saccade_count += 1
                forward_saccade_len += amplitude

        feature_dict['total_saccade_amplitude'] = round(total_saccade_amplitude, 5)
        # 计算平均扫视幅度
        average_saccade_amplitude = round(total_saccade_amplitude / len(saccades), 5)
        feature_dict['average_saccade_amplitude'] = average_saccade_amplitude

        # 计算平均扫视速度
        if total_saccade_duration > 0:
            average_saccade_speed = total_saccade_length / total_saccade_duration
        else:
            average_saccade_speed = 0
        feature_dict['average_saccade_speed'] = round(average_saccade_speed, 5)

        # 计算扫视次数
        saccade_count = len(saccades)
        feature_dict['saccade_count'] = saccade_count

        # 计算扫视朝向方差
        if saccade_angles:
            saccade_angle_variance = np.var(saccade_angles)
        else:
            saccade_angle_variance = 0

        feature_dict['saccade_angle_variance'] = round(saccade_angle_variance, 5)

        # 计算前进扫视比例
        if saccade_count > 0:
            forward_saccade_ratio = round(forward_saccade_count / saccade_count, 5)
        else:
            forward_saccade_ratio = 0

        feature_dict['forward_saccade_ratio'] = forward_saccade_ratio
        feature_dict['avg_forward_saccade_len'] = round(forward_saccade_len / forward_saccade_count, 5)
        feature_dict['avg_backward_saccade_len'] = round(
            (total_saccade_length - forward_saccade_len) / forward_saccade_count, 5)

        return feature_dict
