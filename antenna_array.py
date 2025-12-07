import numpy as np

class AntennaArray2D:
    """
    2D多波束天线阵列类
    
    参数:
        num_antennas_x: x方向天线数量
        num_antennas_y: y方向天线数量
        spacing_x: x方向天线间距（波长倍数）
        spacing_y: y方向天线间距（波长倍数）
        frequency: 工作频率（Hz）
        array_azimuth: 阵列方位角（度），阵列x轴与全局北向的夹角
        array_elevation: 阵列俯仰角（度），阵列平面与水平面的夹角
        array_roll: 阵列滚转角（度），阵列绕自身x轴的旋转角
    """
    def __init__(self, num_antennas_x=8, num_antennas_y=8, spacing_x=0.5, spacing_y=0.5, frequency=1e9,
                 array_azimuth=0.0, array_elevation=0.0, array_roll=0.0):
        self.num_antennas_x = num_antennas_x
        self.num_antennas_y = num_antennas_y
        self.total_antennas = num_antennas_x * num_antennas_y
        self.spacing_x = spacing_x  # 波长倍数
        self.spacing_y = spacing_y  # 波长倍数
        self.frequency = frequency
        self.wavelength = 3e8 / frequency  # 计算波长（m）
        
        # 阵列姿态参数
        self.array_azimuth = array_azimuth  # 阵列方位角（度）
        self.array_elevation = array_elevation  # 阵列俯仰角（度）
        self.array_roll = array_roll  # 阵列滚转角（度）
        
        # 计算天线位置坐标
        self.antenna_positions = self._calculate_antenna_positions()
        
    def _calculate_antenna_positions(self):
        """
        计算2D阵列中每个天线的位置坐标
        返回: 天线位置数组，形状为(total_antennas, 2)，单位为波长
        """
        positions = []
        for y in range(self.num_antennas_y):
            for x in range(self.num_antennas_x):
                pos_x = (x - (self.num_antennas_x - 1) / 2) * self.spacing_x
                pos_y = (y - (self.num_antennas_y - 1) / 2) * self.spacing_y
                positions.append([pos_x, pos_y])
        return np.array(positions)
        
    def get_antenna_positions(self):
        """
        获取天线位置
        """
        return self.antenna_positions
        
    def get_array_parameters(self):
        """
        获取阵列参数
        """
        return {
            'num_antennas_x': self.num_antennas_x,
            'num_antennas_y': self.num_antennas_y,
            'total_antennas': self.total_antennas,
            'spacing_x': self.spacing_x,
            'spacing_y': self.spacing_y,
            'frequency': self.frequency,
            'wavelength': self.wavelength,
            'array_azimuth': self.array_azimuth,
            'array_elevation': self.array_elevation,
            'array_roll': self.array_roll
        }
        
    def set_array_attitude(self, azimuth, elevation, roll):
        """
        设置阵列姿态
        
        参数:
            azimuth: 阵列方位角（度）
            elevation: 阵列俯仰角（度）
            roll: 阵列滚转角（度）
        """
        self.array_azimuth = azimuth
        self.array_elevation = elevation
        self.array_roll = roll
