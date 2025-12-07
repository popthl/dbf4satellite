import numpy as np
from scipy.signal import hilbert
from sgp4.api import Satrec, jday
from sgp4.propagation import gstime
from datetime import datetime

class SatelliteSignal:
    """
    卫星信号生成和处理类
    
    参数:
        frequency: 信号频率（Hz）
        bandwidth: 信号带宽（Hz）
        sampling_rate: 采样率（Hz）
    """
    def __init__(self, frequency=1e9, bandwidth=10e6, sampling_rate=20e6):
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.sampling_rate = sampling_rate
        
    def generate_signal(self, duration, snr_db=20, doppler_shift=0, modulation_type='sinusoid'):
        """
        生成卫星信号，支持多普勒频移和多种调制方式
        
        参数:
            duration: 信号持续时间（秒）
            snr_db: 信噪比（dB）
            doppler_shift: 多普勒频移（Hz）
            modulation_type: 调制类型，可选值：'sinusoid'（正弦波）、'qpsk'（QPSK调制）、'bpsk'（BPSK调制）
        
        返回:
            生成的信号，形状为(N,)
        """
        # 生成时间序列
        t = np.arange(0, duration, 1/self.sampling_rate)
        N = len(t)
        
        # 生成载波信号
        carrier = np.cos(2 * np.pi * (self.frequency + doppler_shift) * t)
        
        # 生成有意义的基带信号
        if modulation_type == 'sinusoid':
            # 正弦波基带信号（简单且容易观察）
            baseband_freq = 100000  # 基带频率100kHz
            baseband = np.cos(2 * np.pi * baseband_freq * t)
        
        elif modulation_type == 'bpsk':
            # BPSK调制信号
            symbol_rate = 10000  # 符号率10kHz
            samples_per_symbol = int(self.sampling_rate / symbol_rate)
            num_symbols = N // samples_per_symbol
            
            # 生成随机比特流
            bits = np.random.randint(0, 2, num_symbols)
            
            # BPSK映射：0→-1，1→1
            symbols = 2 * bits - 1
            
            # 升余弦滤波器
            def raised_cosine_filter(alpha, span, samples_per_symbol):
                t = np.arange(-span/2 * samples_per_symbol, span/2 * samples_per_symbol + 1)
                normalized_t = t / samples_per_symbol
                
                if alpha == 0:
                    return np.sinc(normalized_t)
                else:
                    # 避免除以零
                    denominator1 = np.pi * normalized_t
                    denominator2 = 1 - (2 * alpha * normalized_t)**2
                    
                    # 使用numpy的sinc函数计算sinc项，避免除以零
                    sinc_term = np.sinc(normalized_t)
                    cos_term = np.cos(alpha * np.pi * normalized_t)
                    
                    # 处理分母2为零的情况
                    denominator2 = 1 - (2 * alpha * normalized_t)**2
                    main_term = np.where(denominator2 == 0, 
                                        np.pi/2 * np.sinc(1/(2*alpha)),  # t=±1/(2*alpha)时的极限值
                                        sinc_term * cos_term / denominator2)
                    
                    return main_term
            
            # 生成升余弦脉冲
            alpha = 0.3  # 滚降因子
            span = 4  # 脉冲宽度（符号数）
            pulse = raised_cosine_filter(alpha, span, samples_per_symbol)
            pulse /= np.sqrt(np.sum(pulse**2))  # 能量归一化
            
            # 成型滤波
            baseband = np.zeros(N)
            for i in range(num_symbols):
                start_idx = i * samples_per_symbol
                end_idx = start_idx + len(pulse)
                if end_idx <= N:
                    baseband[start_idx:end_idx] += symbols[i] * pulse
        
        elif modulation_type == 'qpsk':
            # QPSK调制信号
            symbol_rate = 10000  # 符号率10kHz
            samples_per_symbol = int(self.sampling_rate / symbol_rate)
            num_symbols = N // samples_per_symbol
            
            # 生成随机IQ符号
            i_symbols = np.random.randint(0, 2, num_symbols) * 2 - 1  # I路：-1或1
            q_symbols = np.random.randint(0, 2, num_symbols) * 2 - 1  # Q路：-1或1
            
            # 升余弦滤波器
            def raised_cosine_filter(alpha, span, samples_per_symbol):
                t = np.arange(-span/2 * samples_per_symbol, span/2 * samples_per_symbol + 1)
                normalized_t = t / samples_per_symbol
                
                if alpha == 0:
                    return np.sinc(normalized_t)
                else:
                    # 避免除以零
                    denominator1 = np.pi * normalized_t
                    denominator2 = 1 - (2 * alpha * normalized_t)**2
                    
                    # 使用numpy的sinc函数计算sinc项，避免除以零
                    sinc_term = np.sinc(normalized_t)
                    cos_term = np.cos(alpha * np.pi * normalized_t)
                    
                    # 处理分母2为零的情况
                    main_term = np.where(denominator2 == 0, 
                                        np.pi/2 * np.sinc(1/(2*alpha)),  # t=±1/(2*alpha)时的极限值
                                        sinc_term * cos_term / denominator2)
                    
                    return main_term
            
            # 生成升余弦脉冲
            alpha = 0.3  # 滚降因子
            span = 4  # 脉冲宽度（符号数）
            pulse = raised_cosine_filter(alpha, span, samples_per_symbol)
            pulse /= np.sqrt(np.sum(pulse**2))  # 能量归一化
            
            # 成型滤波
            i_baseband = np.zeros(N)
            q_baseband = np.zeros(N)
            for i in range(num_symbols):
                start_idx = i * samples_per_symbol
                end_idx = start_idx + len(pulse)
                if end_idx <= N:
                    i_baseband[start_idx:end_idx] += i_symbols[i] * pulse
                    q_baseband[start_idx:end_idx] += q_symbols[i] * pulse
            
            # 生成QPSK基带信号（I+jQ）
            baseband = i_baseband + 1j * q_baseband
            
        else:
            # 默认使用正弦波
            baseband_freq = 100000  # 基带频率100kHz
            baseband = np.cos(2 * np.pi * baseband_freq * t)
        
        # 调制信号
        # 如果baseband是复数（如QPSK），则使用IQ调制
        if np.iscomplexobj(baseband):
            # IQ调制
            signal = np.real(baseband) * np.cos(2 * np.pi * (self.frequency + doppler_shift) * t) - \
                     np.imag(baseband) * np.sin(2 * np.pi * (self.frequency + doppler_shift) * t)
        else:
            # AM调制
            signal = carrier * baseband
        
        # 计算信号功率
        signal_power = np.mean(np.abs(signal)**2)
        
        # 生成噪声
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(N)
        
        # 添加噪声到信号
        noisy_signal = signal + noise
        
        return noisy_signal
        
    def receive_signal(self, antenna_array, beamformer, directions, duration, snr_db=20):
        """
        模拟天线阵列接收多个卫星信号（支持多个方向和不同多普勒频移，考虑天线姿态）
        
        参数:
            antenna_array: AntennaArray2D对象
            beamformer: Beamformer对象
            directions: 卫星方向列表，每个元素为(azimuth, elevation, doppler_shift)元组
            duration: 接收时间（秒）
            snr_db: 信噪比（dB）
        
        返回:
            每个天线接收的合成信号，形状为(total_antennas, N)
        """
        # 获取天线数量
        total_antennas = antenna_array.total_antennas
        
        # 生成时间序列
        t = np.arange(0, duration, 1/self.sampling_rate)
        received_signals = np.zeros((total_antennas, len(t)), dtype=complex)
        
        # 处理每个方向的信号
        for azimuth, elevation, doppler_shift in directions:
            # 生成带有多普勒频移的信号
            signal = self.generate_signal(duration, snr_db, doppler_shift)
            
            # 将实信号转换为复解析信号
            analytic_signal = hilbert(signal)
            
            # 计算导向矢量（考虑阵列姿态）
            # 利用beamformer的calculate_steering_vector方法，该方法会考虑天线姿态
            # 导向矢量的形式为：steering_vector[i] = np.exp(-1j * phase_delay[i])
            steering_vector = beamformer.calculate_steering_vector(azimuth, elevation, use_array_attitude=True)
            
            # 为每个阵元应用相位延迟并叠加信号
            # 直接使用导向矢量，简化计算：analytic_signal * steering_vector[i] 等价于 analytic_signal * np.exp(-1j * phase_delay[i])
            for i in range(total_antennas):
                received_signals[i, :] += analytic_signal * steering_vector[i]
        
        return received_signals
        
    def process_received_signal(self, received_signals, weights):
        """
        处理接收的信号（应用波束权重）
        
        参数:
            received_signals: 每个天线接收的信号，形状为(total_antennas, N)
            weights: 波束加权系数，形状为(total_antennas,)或(num_beams, total_antennas)
        
        返回:
            处理后的信号，形状为(N,)或(num_beams, N)
        """
        # 检查权重维度
        if weights.ndim == 1:
            # 单波束处理
            processed_signal = np.sum(received_signals * weights[:, np.newaxis], axis=0)
        else:
            # 多波束处理
            num_beams = weights.shape[0]
            processed_signal = np.zeros((num_beams, received_signals.shape[1]), dtype=complex)
            
            for i in range(num_beams):
                processed_signal[i, :] = np.sum(received_signals * weights[i, :, np.newaxis], axis=0)
        
        return processed_signal
        
    def calculate_snr(self, signal, noise):
        """
        计算信噪比
        
        参数:
            signal: 信号
            noise: 噪声
        
        返回:
            信噪比（dB）
        """
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.mean(np.abs(noise)**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return snr_db
        
    def load_tle(self, tle_line1, tle_line2):
        """
        加载TLE数据
        
        参数:
            tle_line1: TLE第一行
            tle_line2: TLE第二行
        
        返回:
            Satrec对象
        """
        satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        return satellite
        
    def calculate_satellite_position(self, satellite, time):
        """
        计算卫星在特定时间的位置和速度
        
        参数:
            satellite: Satrec对象
            time: datetime对象，观测时间
        
        返回:
            卫星位置（km）、速度（km/s）
        """
        # 转换为儒略日
        jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1e6)
        
        # 计算卫星位置和速度
        e, r, v = satellite.sgp4(jd, fr)
        
        if e != 0:
            raise ValueError(f"SGP4计算错误，错误码: {e}")
        
        return r, v
        
    def calculate_azimuth_elevation(self, satellite, ground_lat, ground_lon, ground_alt, time):
        """
        计算地面站相对于卫星的方位角和仰角
        
        参数:
            satellite: Satrec对象
            ground_lat: 地面站纬度（度）
            ground_lon: 地面站经度（度）
            ground_alt: 地面站海拔（km）
            time: datetime对象，观测时间
        
        返回:
            方位角（度）、仰角（度）
        """
        # 转换为儒略日
        jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1e6)
        
        # 计算卫星位置和速度（ECI坐标系）
        e, r_eci, v = satellite.sgp4(jd, fr)
        
        if e != 0:
            raise ValueError(f"SGP4计算错误，错误码: {e}")
        
        # 将ECI坐标转换为ECEF坐标
        # 使用sgp4库内置的gstime函数计算格林尼治恒星时（弧度）
        gst_rad = gstime(jd + fr)
        
        # 计算地球自转矩阵
        cos_gst = np.cos(gst_rad)
        sin_gst = np.sin(gst_rad)
        
        # ECI到ECEF的转换矩阵（简化版，忽略章动和岁差）
        # 适用于地球同步轨道卫星的短期计算
        r_ecef = np.array([
            cos_gst * r_eci[0] + sin_gst * r_eci[1],
            -sin_gst * r_eci[0] + cos_gst * r_eci[1],
            r_eci[2]
        ])
        
        # 地球半径（km）
        R_EARTH = 6378.137
        
        # 地面站地心坐标（km，ECEF坐标系）
        lat_rad = np.radians(ground_lat)
        lon_rad = np.radians(ground_lon)
        
        ground_x = (R_EARTH + ground_alt) * np.cos(lat_rad) * np.cos(lon_rad)
        ground_y = (R_EARTH + ground_alt) * np.cos(lat_rad) * np.sin(lon_rad)
        ground_z = (R_EARTH + ground_alt) * np.sin(lat_rad)
        ground_pos = np.array([ground_x, ground_y, ground_z])
        
        # 卫星相对于地面站的向量（km）
        rel_pos = r_ecef - ground_pos
        
        # 计算距离
        distance = np.linalg.norm(rel_pos)
        
        # 计算仰角
        # 卫星在地面站坐标系中的高度角
        # 地面站本地坐标系：x-东，y-北，z-天顶
        
        # 计算地面站的单位向量
        # 东向单位向量
        east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
        # 北向单位向量
        north = np.array([-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)])
        # 天顶单位向量
        zenith = np.array([np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)])
        
        # 计算卫星在地面站本地坐标系中的坐标
        sat_local_x = np.dot(rel_pos, east)
        sat_local_y = np.dot(rel_pos, north)
        sat_local_z = np.dot(rel_pos, zenith)
        
        # 计算仰角（度）
        elevation = np.degrees(np.arcsin(sat_local_z / distance))
        
        # 计算方位角（度）
        azimuth = np.degrees(np.arctan2(sat_local_x, sat_local_y))
        if azimuth < 0:
            azimuth += 360
        
        return azimuth, elevation
        
    def calculate_doppler_shift(self, satellite, ground_lat, ground_lon, ground_alt, time):
        """
        计算地面站接收到的卫星信号多普勒频移
        
        参数:
            satellite: Satrec对象
            ground_lat: 地面站纬度（度）
            ground_lon: 地面站经度（度）
            ground_alt: 地面站海拔（km）
            time: datetime对象，观测时间
        
        返回:
            多普勒频移（Hz）
        """
        # 转换为儒略日
        jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1e6)
        
        # 计算卫星位置和速度（ECI坐标系）
        e, r_eci, v_eci = satellite.sgp4(jd, fr)
        
        if e != 0:
            raise ValueError(f"SGP4计算错误，错误码: {e}")
        
        # 将ECI坐标转换为ECEF坐标
        # 使用sgp4库内置的gstime函数计算格林尼治恒星时（弧度）
        gst_rad = gstime(jd + fr)
        
        # 计算地球自转矩阵
        cos_gst = np.cos(gst_rad)
        sin_gst = np.sin(gst_rad)
        
        # ECI到ECEF的转换矩阵（简化版，忽略章动和岁差）
        r_ecef = np.array([
            cos_gst * r_eci[0] + sin_gst * r_eci[1],
            -sin_gst * r_eci[0] + cos_gst * r_eci[1],
            r_eci[2]
        ])
        
        # 速度也要进行ECI到ECEF的转换
        v_ecef = np.array([
            cos_gst * v_eci[0] + sin_gst * v_eci[1],
            -sin_gst * v_eci[0] + cos_gst * v_eci[1],
            v_eci[2]
        ])
        
        # 地球半径（km）
        R_EARTH = 6378.137
        
        # 地面站地心坐标（km，ECEF坐标系）
        lat_rad = np.radians(ground_lat)
        lon_rad = np.radians(ground_lon)
        
        ground_x = (R_EARTH + ground_alt) * np.cos(lat_rad) * np.cos(lon_rad)
        ground_y = (R_EARTH + ground_alt) * np.cos(lat_rad) * np.sin(lon_rad)
        ground_z = (R_EARTH + ground_alt) * np.sin(lat_rad)
        ground_pos = np.array([ground_x, ground_y, ground_z])
        
        # 卫星相对于地面站的向量（km）
        rel_pos = r_ecef - ground_pos
        
        # 计算距离（km）
        distance = np.linalg.norm(rel_pos)
        
        # 计算卫星相对于地面站的速度分量（km/s）
        # 速度投影到视线方向的分量
        rel_vel = v_ecef
        # 视线方向单位向量（从地面站到卫星）
        los_unit = rel_pos / distance
        # 视线方向的速度分量（km/s）
        # 注意：这里是卫星相对于地面站的速度，所以如果卫星远离地面站，这个值会是正的
        los_vel = np.dot(rel_vel, los_unit)
        
        # 光速（km/s）
        c = 299792.458
        
        # 计算多普勒频移（Hz）
        # 频移公式：delta_f = -f0 * v / c
        # 其中v是相对速度，当卫星靠近时v为负，所以频移为正
        doppler_shift = -self.frequency * los_vel / c
        
        return doppler_shift
        
    def get_satellite_signal_parameters(self, tle_line1, tle_line2, ground_lat, ground_lon, ground_alt, time, elevation_threshold=5):
        """
        根据TLE文件和地面站位置，计算卫星信号的参数
        
        参数:
            tle_line1: TLE第一行
            tle_line2: TLE第二行
            ground_lat: 地面站纬度（度）
            ground_lon: 地面站经度（度）
            ground_alt: 地面站海拔（km）
            time: datetime对象，观测时间
            elevation_threshold: 仰角门限（度），低于此值则认为无法接收信号
        
        返回:
            字典，包含以下信息：
            - azimuth: 方位角（度）
            - elevation: 仰角（度）
            - doppler_shift: 多普勒频移（Hz）
            - can_receive: 是否可以接收信号（仰角>=门限）
        """
        # 加载TLE数据
        satellite = self.load_tle(tle_line1, tle_line2)
        
        # 计算方位角和仰角
        azimuth, elevation = self.calculate_azimuth_elevation(satellite, ground_lat, ground_lon, ground_alt, time)
        
        # 计算多普勒频移
        doppler_shift = self.calculate_doppler_shift(satellite, ground_lat, ground_lon, ground_alt, time)
        
        # 判断是否可以接收信号
        can_receive = elevation >= elevation_threshold
        
        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'doppler_shift': doppler_shift,
            'can_receive': can_receive
        }
