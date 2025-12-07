import numpy as np
from antenna_array import AntennaArray2D

class Beamformer:
    """
    2D多波束形成器类
    
    参数:
        antenna_array: AntennaArray2D对象
    """
    def __init__(self, antenna_array):
        self.antenna_array = antenna_array
        
    def _euler_rotation(self, azimuth, elevation, array_az, array_el, array_roll):
        """
        应用欧拉旋转，将全局坐标系下的目标方向转换为阵列本地坐标系下的方向
        
        参数:
            azimuth: 目标全局方位角（度）
            elevation: 目标全局俯仰角（度）
            array_az: 阵列方位角（度）
            array_el: 阵列俯仰角（度）
            array_roll: 阵列滚转角（度）
        
        返回:
            阵列本地坐标系下的方位角和俯仰角（度）
        """
        # 将角度转换为弧度
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        array_az_rad = np.radians(array_az)
        array_el_rad = np.radians(array_el)
        array_roll_rad = np.radians(array_roll)
        
        # 将目标方向转换为单位向量（全局坐标系）
        # 全局坐标系：x=东，y=北，z=天
        global_x = np.sin(el_rad) * np.sin(az_rad)
        global_y = np.sin(el_rad) * np.cos(az_rad)
        global_z = np.cos(el_rad)
        
        # 旋转矩阵：Z-Y-X欧拉角旋转（方位角-俯仰角-滚转角）
        # 1. 绕Z轴旋转-array_az（将全局坐标系旋转到阵列方位）
        Rz = np.array([
            [np.cos(-array_az_rad), -np.sin(-array_az_rad), 0],
            [np.sin(-array_az_rad), np.cos(-array_az_rad), 0],
            [0, 0, 1]
        ])
        
        # 2. 绕Y轴旋转-array_el（将全局坐标系旋转到阵列俯仰）
        Ry = np.array([
            [np.cos(-array_el_rad), 0, np.sin(-array_el_rad)],
            [0, 1, 0],
            [-np.sin(-array_el_rad), 0, np.cos(-array_el_rad)]
        ])
        
        # 3. 绕X轴旋转-array_roll（将全局坐标系旋转到阵列滚转）
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(-array_roll_rad), -np.sin(-array_roll_rad)],
            [0, np.sin(-array_roll_rad), np.cos(-array_roll_rad)]
        ])
        
        # 组合旋转矩阵：先Z，再Y，最后X
        R = Rx @ Ry @ Rz
        
        # 将全局单位向量转换为阵列本地坐标系下的单位向量
        global_vec = np.array([global_x, global_y, global_z])
        local_vec = R @ global_vec
        
        # 从本地单位向量转换为方位角和俯仰角
        local_el = np.arccos(local_vec[2])  # 俯仰角：与z轴的夹角
        local_az = np.arctan2(local_vec[0], local_vec[1])  # 方位角：x-y平面内与y轴的夹角
        
        # 转换回角度
        local_az_deg = np.degrees(local_az)
        local_el_deg = np.degrees(local_el)
        
        return local_az_deg, local_el_deg
        
    def calculate_steering_vector(self, azimuth, elevation, use_array_attitude=True):
        """
        计算导向矢量
        
        参数:
            azimuth: 方位角（度），范围[-180, 180]
            elevation: 俯仰角（度），范围[0, 90]
            use_array_attitude: 是否考虑阵列姿态
        
        返回:
            导向矢量，形状为(total_antennas,)
        """
        # 获取阵列姿态参数
        array_az = self.antenna_array.array_azimuth
        array_el = self.antenna_array.array_elevation
        array_roll = self.antenna_array.array_roll
        
        # 如果考虑阵列姿态，进行坐标转换
        if use_array_attitude:
            # _euler_rotation 返回的是度数，需要转换为弧度
            local_az_deg, local_el_deg = self._euler_rotation(azimuth, elevation, array_az, array_el, array_roll)
            az_rad = np.radians(local_az_deg)
            el_rad = np.radians(local_el_deg)
        else:
            # 直接使用输入角度，转换为弧度
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
        
        # 计算波数向量在x和y方向的分量（单位为2π/波长）
        # 注意：positions中的单位是波长，所以2π/λ * λ = 2π，因此可以简化计算
        kx = np.sin(el_rad) * np.cos(az_rad)
        ky = np.sin(el_rad) * np.sin(az_rad)
        
        # 获取天线位置（单位为波长）
        positions = self.antenna_array.get_antenna_positions()
        
        # 计算相位延迟（考虑波数向量和阵元位置）
        # 相位延迟 = k · r，其中k是波数向量（2π/λ），r是阵元位置
        # 由于positions的单位是波长，所以相位延迟 = 2π * (kx * x + ky * y)
        phase_delay = 2 * np.pi * (kx * positions[:, 0] + ky * positions[:, 1])
        
        # 导向矢量
        steering_vector = np.exp(-1j * phase_delay)
        
        return steering_vector
        
    def calculate_array_factor(self, azimuth, elevation, weights=None):
        """
        计算阵列因子
        
        参数:
            azimuth: 方位角（度）
            elevation: 俯仰角（度）
            weights: 加权系数，形状为(total_antennas,)，默认为均匀加权
        
        返回:
            阵列因子（复数）
        """
        if weights is None:
            weights = np.ones(self.antenna_array.total_antennas)
        
        steering_vector = self.calculate_steering_vector(azimuth, elevation)
        array_factor = np.sum(weights * steering_vector)
        
        return array_factor
        
    def calculate_beam_pattern(self, az_range, el_range, weights=None):
        """
        计算波束方向图
        
        参数:
            az_range: 方位角范围，形状为(N,)（度）
            el_range: 俯仰角范围，形状为(M,)（度）
            weights: 加权系数，形状为(total_antennas,)，默认为均匀加权
        
        返回:
            波束方向图，形状为(M, N)，单位为dB
        """
        if weights is None:
            weights = np.ones(self.antenna_array.total_antennas)
        
        # 创建网格
        AZ, EL = np.meshgrid(az_range, el_range)
        pattern = np.zeros(AZ.shape, dtype=complex)
        
        # 计算每个方向的阵列因子
        for i in range(AZ.shape[0]):
            for j in range(AZ.shape[1]):
                pattern[i, j] = self.calculate_array_factor(AZ[i, j], EL[i, j], weights)
        
        # 转换为dB
        pattern_abs = np.abs(pattern)
        pattern_abs = np.maximum(pattern_abs, 1e-10)  # 避免除以零
        pattern_dB = 20 * np.log10(pattern_abs / np.max(pattern_abs))
        
        return pattern_dB
        
    def get_beam_weights(self, target_az, target_el):
        """
        计算指向特定方向的波束加权系数（基于最大比合并）
        
        参数:
            target_az: 目标方位角（度）
            target_el: 目标俯仰角（度）
        
        返回:
            加权系数，形状为(total_antennas,)
        """
        # 计算指向目标方向的导向矢量
        steering_vector = self.calculate_steering_vector(target_az, target_el)
        
        # 最大比合并权重（共轭匹配）
        weights = np.conj(steering_vector)
        
        return weights
        
    def get_multiple_beam_weights(self, target_directions):
        """
        计算指向多个目标方向的波束加权系数
        
        参数:
            target_directions: 目标方向列表，每个元素为(az, el)元组
        
        返回:
            加权系数矩阵，形状为(len(target_directions), total_antennas)
        """
        weights = []
        for az, el, doppler in target_directions:
            w = self.get_beam_weights(az, el)
            weights.append(w)
        return np.array(weights)
