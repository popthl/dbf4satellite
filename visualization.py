import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

class Visualization:
    def __init__(self):
        """初始化可视化类"""
        # 配置matplotlib参数
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题，使用ASCII负号
        plt.rcParams['axes.formatter.use_mathtext'] = False  # 禁用mathtext，避免使用Unicode减号
    
    def plot_array_geometry(self, positions, title="Antenna Array Geometry"):
        """
        绘制天线阵列的几何布局
        
        参数:
        positions: np.ndarray - 天线位置数组，形状为(N, 2)，单位为波长
        title: str - 图形标题
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(positions[:, 0], positions[:, 1], s=100, c='b', marker='o')
        
        # 标记天线编号
        for i, (x, y) in enumerate(positions):
            plt.text(x, y, f'{i+1}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        
        plt.xlabel('X Position (λ)')
        plt.ylabel('Y Position (λ)')
        plt.title(title)
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
    
    def plot_beam_pattern_2d(self, az_range, el_range, pattern_dB, title="Beam Pattern (2D)"):
        """
        绘制2D波束方向图
        
        参数:
        az_range: np.ndarray - 方位角范围，单位为度
        el_range: np.ndarray - 俯仰角范围，单位为度
        pattern_dB: np.ndarray - 波束方向图，dB值，形状为(len(el_range), len(az_range))
        title: str - 图形标题
        """
        plt.figure(figsize=(10, 8))
        
        # 转换为网格
        az_grid, el_grid = np.meshgrid(az_range, el_range)
        
        # 绘制等高线图
        contour = plt.contourf(az_grid, el_grid, pattern_dB, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Normalized Power (dB)')
        
        plt.xlabel('Azimuth Angle (°)')
        plt.ylabel('Elevation Angle (°)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
    
    def plot_beam_pattern_3d(self, az_range, el_range, pattern_dB, title="Beam Pattern (3D)"):
        """
        绘制3D波束方向图
        
        参数:
        az_range: np.ndarray - 方位角范围，单位为度
        el_range: np.ndarray - 俯仰角范围，单位为度
        pattern_dB: np.ndarray - 波束方向图，dB值，形状为(len(el_range), len(az_range))
        title: str - 图形标题
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 转换为网格
        az_grid, el_grid = np.meshgrid(az_range, el_range)
        
        # 绘制3D表面图
        surf = ax.plot_surface(az_grid, el_grid, pattern_dB, 
                              cmap='viridis', edgecolor='none', alpha=0.8)
        fig.colorbar(surf, label='Normalized Power (dB)')
        
        ax.set_xlabel('Azimuth Angle (°)')
        ax.set_ylabel('Elevation Angle (°)')
        ax.set_zlabel('Normalized Power (dB)')
        ax.set_title(title)
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
    
    def plot_beam_pattern_hemisphere(self, az_range, el_range, pattern_dB, title="Beam Pattern (Hemisphere)"):
        """
        在半球面上绘制波束方向图
        
        参数:
        az_range: np.ndarray - 方位角范围，单位为度
        el_range: np.ndarray - 俯仰角范围，单位为度
        pattern_dB: np.ndarray - 波束方向图，dB值，形状为(len(el_range), len(az_range))
        title: str - 图形标题
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 将角度转换为弧度
        az_rad = np.radians(az_range)
        el_rad = np.radians(el_range)
        
        # 创建网格
        az_grid, el_grid = np.meshgrid(az_rad, el_rad)
        
        # 转换为球面坐标 (r, theta, phi) 到笛卡尔坐标 (x, y, z)
        # 其中：
        # - r 是归一化的功率（转换为线性刻度）
        # - theta 是俯仰角（从z轴向下测量）
        # - phi 是方位角（绕z轴旋转）
        r = 10**(pattern_dB / 20)  # 转换回线性刻度
        r = r / np.max(r)  # 归一化到0-1范围
        
        x = r * np.sin(el_grid) * np.cos(az_grid)
        y = r * np.sin(el_grid) * np.sin(az_grid)
        z = r * np.cos(el_grid)
        
        # 绘制半球面波束方向图
        surf = ax.plot_surface(x, y, z, 
                              cmap='viridis', edgecolor='none', alpha=0.8)
        fig.colorbar(surf, label='Normalized Power (dB)')
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 设置等比例坐标轴
        ax.set_aspect('equal')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
    
    def plot_beam_pattern_polar(self, az_range, pattern_slice, title="Polar Beam Pattern"):
        """
        绘制极坐标波束方向图
        
        参数:
        az_range: np.ndarray - 方位角范围，单位为度
        pattern_slice: np.ndarray - 特定俯仰角下的波束方向图切片，dB值
        title: str - 图形标题
        """
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection='polar')
        
        # 转换方位角为弧度
        az_rad = np.deg2rad(az_range)
        
        # 绘制极坐标图
        ax.plot(az_rad, pattern_slice, linewidth=2, label='Beam Pattern')
        ax.fill_between(az_rad, pattern_slice, np.min(pattern_slice), alpha=0.3)
        
        # 设置极坐标参数
        ax.set_theta_zero_location('N')  # 0度在正北方向
        ax.set_theta_direction(-1)  # 顺时针方向
        ax.set_rlabel_position(225)  # 半径标签位置
        ax.set_ylim(np.min(pattern_slice), 0)
        
        plt.title(title)
        plt.tight_layout()
    
    def plot_signal_time_domain(self, signal, sampling_rate, 
                              title="Received Signal (Time Domain)"):
        """
        绘制时域信号
        
        参数:
        signal: np.ndarray - 接收信号
        sampling_rate: float - 采样率
        title: str - 图形标题
        """
        plt.figure(figsize=(12, 6))
        
        # 生成时间轴
        time = np.arange(len(signal)) / sampling_rate
        
        plt.plot(time, np.real(signal), linewidth=1, label='Real Part')
        #plt.plot(time, np.imag(signal), linewidth=1, label='Imaginary Part')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        #plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    def plot_signal_comparison(self, single_element_signal, beamformed_signal, sampling_rate, 
                              title="Signal Comparison"):
        """
        对比单个阵元信号和波束形成后的信号
        
        参数:
        single_element_signal: np.ndarray - 单个阵元接收的信号
        beamformed_signal: np.ndarray - 波束形成后的信号
        sampling_rate: float - 采样率
        title: str - 图形标题
        """
        plt.figure(figsize=(12, 10))
        
        # 生成时间轴
        time = np.arange(len(single_element_signal)) / sampling_rate
        
        # 绘制单个阵元信号（时域）
        plt.subplot(2, 1, 1)
        plt.plot(time, np.real(single_element_signal), linewidth=1, label='Single Element - Real')
        #plt.plot(time, np.imag(single_element_signal), linewidth=1, label='Single Element - Imaginary')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'{title} - Single Element Signal')
        #plt.legend()
        plt.grid(True)
        
        # 绘制波束形成后的信号（时域）
        plt.subplot(2, 1, 2)
        plt.plot(time, np.real(beamformed_signal), linewidth=1, label='Beamformed - Real')
        #plt.plot(time, np.imag(beamformed_signal), linewidth=1, label='Beamformed - Imaginary')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'{title} - Beamformed Signal')
        #plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
    
    def plot_signal_spectrum_comparison(self, single_element_signal, beamformed_signal, sampling_rate, 
                                      title="Signal Spectrum Comparison"):
        """
        对比单个阵元信号和波束形成后的信号频谱
        
        参数:
        single_element_signal: np.ndarray - 单个阵元接收的信号
        beamformed_signal: np.ndarray - 波束形成后的信号
        sampling_rate: float - 采样率
        title: str - 图形标题
        """
        plt.figure(figsize=(12, 8))
        
        # 计算FFT
        n = len(single_element_signal)
        freq = np.fft.fftfreq(n, 1/sampling_rate)
        
        # 只显示正频率部分
        positive_freq_mask = freq >= 0
        freq = freq[positive_freq_mask]
        
        # 单个阵元信号频谱
        single_spectrum = np.abs(np.fft.fft(single_element_signal)) / n
        single_spectrum = single_spectrum[positive_freq_mask]
        
        # 波束形成后的信号频谱
        beamformed_spectrum = np.abs(np.fft.fft(beamformed_signal)) / n
        beamformed_spectrum = beamformed_spectrum[positive_freq_mask]
        
        # 转换为dB
        single_spectrum_dB = 20 * np.log10(np.maximum(single_spectrum, 1e-10))
        beamformed_spectrum_dB = 20 * np.log10(np.maximum(beamformed_spectrum, 1e-10))
        
        # 绘制频谱对比
        plt.plot(freq, single_spectrum_dB, linewidth=1, label='Single Element')
        plt.plot(freq, beamformed_spectrum_dB, linewidth=1, label='Beamformed')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    def plot_all_elements_time_domain(self, received_signals, sampling_rate, time_range_us=2, 
                                     title="All Elements Time Domain Signal"):
        """
        绘制所有阵元的时域信号曲线，只显示指定时间范围内的数据
        
        参数:
        received_signals: np.ndarray - 所有阵元接收的信号，形状为(total_antennas, N)
        sampling_rate: float - 采样率
        time_range_us: float - 显示的时间范围（微秒）
        title: str - 图形标题
        """
        # 计算时间范围内的采样点数量
        num_samples = int(sampling_rate * time_range_us * 1e-6)
        
        # 只取前num_samples个采样点
        signals_short = received_signals[:, :num_samples]
        
        # 生成时间轴（微秒）
        time = np.arange(num_samples) / sampling_rate * 1e6
        
        # 创建图形
        plt.figure(figsize=(14, 10))
        
        # 绘制所有阵元的实部信号
        for i in range(signals_short.shape[0]):
            if i%8 != 0:
                continue
            print(f'Element {i}')
            plt.plot(time, np.real(signals_short[i, :]), linewidth=0.8, alpha=0.7, label=f'Element {i}')
        
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude (Real Part)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        
        # 添加图例（只显示部分阵元，避免图例过多）
        if signals_short.shape[0] <= 10:
            plt.legend()
        else:
            plt.legend([f'Element {i}' for i in range(5)] + ['...', f'Element {signals_short.shape[0]-1}'])
    
    def plot_all_elements_time_domain_separate(self, received_signals, sampling_rate, time_range_us=2, 
                                             title="All Elements Time Domain Signal (Separate)"):
        """
        绘制所有阵元的时域信号曲线，每个阵元单独一行显示
        
        参数:
        received_signals: np.ndarray - 所有阵元接收的信号，形状为(total_antennas, N)
        sampling_rate: float - 采样率
        time_range_us: float - 显示的时间范围（微秒）
        title: str - 图形标题
        """
        # 计算时间范围内的采样点数量
        num_samples = int(sampling_rate * time_range_us * 1e-6)
        
        # 只取前num_samples个采样点
        signals_short = received_signals[:, :num_samples]
        
        # 生成时间轴（微秒）
        time = np.arange(num_samples) / sampling_rate * 1e6
        
        # 获取阵元数量
        total_antennas = signals_short.shape[0]
        
        # 创建图形，每行显示8个阵元
        rows = (total_antennas + 7) // 8
        plt.figure(figsize=(14, 2 * rows))
        
        for i in range(total_antennas):
            plt.subplot(rows, 8, i+1)
            plt.plot(time, np.real(signals_short[i, :]), linewidth=1)
            plt.title(f'Element {i}')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
    
    def plot_element_phase_differences(self, received_signals, reference_element=0, 
                                     title="Element Phase Differences"):
        """
        绘制各个阵元相对于参考阵元的相位差
        
        参数:
        received_signals: np.ndarray - 所有阵元接收的信号，形状为(total_antennas, N)
        reference_element: int - 参考阵元的索引
        title: str - 图形标题
        """
        # 获取阵元数量
        total_antennas = received_signals.shape[0]
        
        # 使用信号的前1000个采样点进行计算，减少计算量
        signal_length = min(1000, received_signals.shape[1])
        
        # 计算每个阵元与参考阵元的相位差
        phase_diffs = []
        ref_signal = received_signals[reference_element, :signal_length]
        
        for i in range(total_antennas):
            # 使用互相关计算相位差
            corr = np.correlate(ref_signal, received_signals[i, :signal_length], mode='full')
            # 找到最大相关值的位置
            max_corr_idx = np.argmax(corr)
            # 计算时间延迟（采样点）
            time_delay = max_corr_idx - (signal_length - 1)
            # 转换为相位差（弧度）
            # 假设信号频率为1GHz，周期为1ns
            phase_diff = (time_delay / signal_length) * 2 * np.pi
            phase_diffs.append(phase_diff)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制相位差
        plt.plot(range(total_antennas), phase_diffs, 'o-', linewidth=2, markersize=8)
        
        plt.xlabel('Element Index')
        plt.ylabel('Phase Difference (radians)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        
        # 添加参考线
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label=f'Reference: Element {reference_element}')
        plt.legend()
    
    def plot_phase_distribution_2d(self, received_signals, num_antennas_x=8, num_antennas_y=8, 
                                  reference_element=0, title="Phase Distribution (2D)"):
        """
        绘制2D阵列的相位差分布
        
        参数:
        received_signals: np.ndarray - 所有阵元接收的信号，形状为(total_antennas, N)
        num_antennas_x: int - x方向的阵元数量
        num_antennas_y: int - y方向的阵元数量
        reference_element: int - 参考阵元的索引
        title: str - 图形标题
        """
        # 获取阵元数量
        total_antennas = received_signals.shape[0]
        
        # 使用信号的前1000个采样点进行计算，减少计算量
        signal_length = min(1000, received_signals.shape[1])
        
        # 计算每个阵元与参考阵元的相位差
        phase_diffs = []
        ref_signal = received_signals[reference_element, :signal_length]
        
        for i in range(total_antennas):
            # 使用互相关计算相位差
            corr = np.correlate(ref_signal, received_signals[i, :signal_length], mode='full')
            max_corr_idx = np.argmax(corr)
            time_delay = max_corr_idx - (signal_length - 1)
            phase_diff = (time_delay / signal_length) * 2 * np.pi
            phase_diffs.append(phase_diff)
        
        # 将相位差转换为2D矩阵
        phase_matrix = np.array(phase_diffs).reshape(num_antennas_y, num_antennas_x)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制2D相位分布图
        im = plt.imshow(phase_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Phase Difference (radians)')
        
        # 添加阵元编号
        for i in range(num_antennas_y):
            for j in range(num_antennas_x):
                plt.text(j, i, f'{i*num_antennas_x + j}', ha='center', va='center', 
                        color='white', fontsize=8, fontweight='bold')
        
        plt.xlabel('X Element Index')
        plt.ylabel('Y Element Index')
        plt.title(title)
        plt.grid(False)
        plt.tight_layout()
    
    def plot_signal_frequency_domain(self, signal, sampling_rate, title="Received Signal (Frequency Domain)"):
        """
        绘制频域信号
        
        参数:
        signal: np.ndarray - 接收信号
        sampling_rate: float - 采样率
        title: str - 图形标题
        """
        plt.figure(figsize=(12, 6))
        
        # 计算FFT
        n = len(signal)
        freq = np.fft.fftfreq(n, 1/sampling_rate)
        spectrum = np.abs(np.fft.fft(signal)) / n
        
        # 只显示正频率部分
        positive_freq_mask = freq >= 0
        freq = freq[positive_freq_mask]
        spectrum = spectrum[positive_freq_mask]
        
        plt.plot(freq, 20 * np.log10(np.maximum(spectrum, 1e-10)), linewidth=1)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
    
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
        global_x = np.cos(el_rad) * np.sin(az_rad)
        global_y = np.cos(el_rad) * np.cos(az_rad)
        global_z = np.sin(el_rad)
        
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
        #print(global_vec,local_vec)
        # 从本地单位向量转换为方位角和俯仰角
        local_el = np.arcsin(local_vec[2])  # 俯仰角：与z轴的夹角
        local_az = np.arctan2(local_vec[0], local_vec[1])  # 方位角：x-y平面内与y轴的夹角
        
        # 转换回角度
        local_az_deg = np.degrees(local_az)
        local_el_deg = np.degrees(local_el)
        
        return local_az_deg, local_el_deg
    
    def plot_sky_plot(self, satellite_data, title="Satellite Sky Plot", max_labels=40, array_attitude=None, elevation_threshold=5):
        """
        绘制地面站可见卫星的天空图
        
        参数:
        satellite_data: 字典列表，每个字典包含卫星名称、方位角(azimuth)和仰角(elevation)
        title: str - 图形标题
        max_labels: int - 最多显示的标签数量（默认40）
        array_attitude: 字典，包含阵列姿态参数：azimuth, elevation, roll（度）
                      如果提供，则将卫星位置转换为阵列本地坐标系
        elevation_threshold: float - 可见性门限（度），本地仰角大于等于此值才可见
        """
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111, projection='polar')
        
        # 设置极坐标参数
        ax.set_theta_zero_location('N')  # 0度在正北方向
        ax.set_theta_direction(-1)  # 顺时针方向
        ax.set_rlim(0, 90)  # 径向范围从0到90度（仰角）
        ax.set_rlabel_position(135)  # 半径标签位置
        
        # 反转径向轴，使得90度（天顶）在中心，0度（地平线）在外围
        ax.set_yticks([0, 5, 30, 60, 90])
        ax.set_yticklabels(['0°', '5°','30°', '60°', '90°'])
        ax.set_ylim(90, 0)  # 反转径向轴
        
        # 绘制同心圆（仰角线）
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 标记方位角方向
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        
        # 处理所有卫星，计算本地坐标并重新判断可见性
        processed_satellites = []
        
        for sat in satellite_data:
            global_az = sat['azimuth']
            global_el = sat['elevation']
            doppler_shift = sat['doppler_shift']
            
            # 转换为本地坐标系
            if array_attitude:
                local_az, local_el = self._euler_rotation(
                    global_az, global_el,
                    array_attitude['azimuth'],
                    array_attitude['elevation'],
                    array_attitude['roll']
                )
            else:
                local_az, local_el = global_az, global_el
            
            # 正确的可见性判断逻辑：
            # 1. 全局仰角 > 0°（卫星真的在地平线以上）
            # 2. 并且本地仰角 >= 门限（卫星在天线视野范围内）
            is_visible = (global_el > 0) and (local_el >= elevation_threshold)
            
            processed_satellites.append({
                'sat': sat,
                'local_az': local_az,
                'local_el': local_el,
                'is_visible': is_visible,
                'doppler_shift': doppler_shift
            })
        
        # 分离可见和不可见卫星
        visible_satellites = [p for p in processed_satellites if p['is_visible']]
        invisible_satellites = [p for p in processed_satellites if not p['is_visible']]
        print(f"总卫星数: {len(processed_satellites)}")
        print(f"可见卫星数: {len(visible_satellites)}")
        print(f"不可见卫星数: {len(invisible_satellites)}")
        
        # 绘制不可见卫星（灰色，较小）
        if invisible_satellites:
            az_invisible = np.deg2rad([p['local_az'] for p in invisible_satellites])
            el_invisible = [p['local_el'] for p in invisible_satellites]
            #sv_invisible = [p['sat']['name'] for p in invisible_satellites]
            ax.scatter(az_invisible, el_invisible, s=30, c='gray', marker='o', alpha=0.5, label='不可见卫星')
            #print(f"不可见卫星名称: {sv_invisible}")
            #print(f"不可见卫星方位角: {az_invisible}")
            #print(f"不可见卫星仰角: {el_invisible}")

        
        # 绘制可见卫星（彩色，较大）
        if visible_satellites:
            az_visible = np.deg2rad([p['local_az'] for p in visible_satellites])
            el_visible = [p['local_el'] for p in visible_satellites]
            doppler_visible = [p['doppler_shift'] for p in visible_satellites]
            
            # 绘制卫星，颜色表示多普勒频移
            scatter = ax.scatter(az_visible, el_visible, s=80, c=doppler_visible, cmap='coolwarm', 
                               marker='o', alpha=0.8, label='可见卫星')
            
            # 添加颜色条，显示多普勒频移
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('多普勒频移 (Hz)')
            
            # 为可见卫星添加标签
            for i, p in enumerate(visible_satellites[:max_labels]):
                sat = p['sat']
                az_rad = np.deg2rad(p['local_az'])
                el = p['local_el']
                doppler = p['doppler_shift']
                
                # 提取卫星编号（如C06）
                sat_name = sat['name']
                # 查找括号中的编号，如(C06)
                if '(' in sat_name and ')' in sat_name:
                    # 提取括号内的内容
                    sat_id = sat_name[sat_name.find('(')+1:sat_name.find(')')]
                else:
                    # 如果没有括号，使用原始名称的前5个字符
                    sat_id = sat_name[:5]
                
                ax.text(az_rad, el, 
                       sat_id, ha='center', va='center', fontsize=8)#, 
                       #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 如果有更多可见卫星没有显示标签，添加说明
            if len(visible_satellites) > max_labels:
                print(f"提示：共有 {len(visible_satellites)} 颗可见卫星，仅显示前 {max_labels} 个标签以避免重叠。")
                print("可以通过调整 max_labels 参数来显示更多标签。")
        
        # 如果提供了阵列姿态，添加阵列方向指示
        if array_attitude:
            # 绘制阵列指向方向（本地坐标系中的天顶方向）
            # 在全局坐标系中，阵列指向是一个方向向量，经过旋转后在本地坐标系中指向天顶
            nue_az = [0,0,90]
            nue_el = [0,90,0]
            lnue_az, lnue_el = self._euler_rotation(
                    nue_az, nue_el,
                    array_attitude['azimuth'],
                    array_attitude['elevation'],
                    array_attitude['roll']
                )
            ax.plot(np.deg2rad([0, lnue_az[0]]), [90, lnue_el[0]], color='red', linewidth=1, label='N')
            ax.plot(np.deg2rad([0, lnue_az[1]]), [90, lnue_el[1]], color='green', linewidth=1, label='U')
            ax.plot(np.deg2rad([0, lnue_az[2]]), [90, lnue_el[2]], color='blue', linewidth=1, label='E')
            ax.scatter(np.deg2rad(lnue_az), lnue_el, s=200, color='red', marker='+')
        
        plt.title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
    
    def plot_sky_plot_with_trajectory(self, satellite_trajectories, title="Satellite Sky Plot with Trajectory"):
        """
        绘制一段时间内的天空图，每颗卫星形成一条轨迹
        
        参数:
        satellite_trajectories: 字典，键为卫星名称，值为包含时间、方位角、仰角、多普勒频移和接收状态的列表
        title: str - 图形标题
        """
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111, projection='polar')
        
        # 设置极坐标参数
        ax.set_theta_zero_location('N')  # 0度在正北方向
        ax.set_theta_direction(-1)  # 顺时针方向
        ax.set_rlim(0, 90)  # 径向范围从0到90度（仰角）
        ax.set_rlabel_position(135)  # 半径标签位置
        
        # 反转径向轴，使得90度（天顶）在中心，0度（地平线）在外围
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(['0°', '30°', '60°', '90°'])
        ax.set_ylim(90, 0)  # 反转径向轴
        
        # 绘制同心圆（仰角线）
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 标记方位角方向
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        
        # 收集所有可见点数据，用于颜色映射
        all_visible_points = []
        all_doppler = []
        
        # 遍历所有卫星轨迹，收集可见点
        for sat_name, trajectory in satellite_trajectories.items():
            for point in trajectory:
                if point['can_receive']:
                    all_visible_points.append((sat_name, point))
                    all_doppler.append(point['doppler_shift'])
        
        # 绘制所有可见点，颜色表示多普勒频移
        if all_visible_points:
            # 提取所有可见点的方位角、仰角和多普勒频移
            az_rad = np.deg2rad([p[1]['azimuth'] for p in all_visible_points])
            el = [p[1]['elevation'] for p in all_visible_points]
            doppler = [p[1]['doppler_shift'] for p in all_visible_points]
            
            # 绘制点，颜色表示多普勒频移
            scatter = ax.scatter(az_rad, el, s=50, c=doppler, cmap='coolwarm', 
                               alpha=0.8, marker='o')
            
            # 添加颜色条，显示多普勒频移
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('多普勒频移 (Hz)')
        
        # 添加卫星标签
        for sat_name, trajectory in satellite_trajectories.items():
            # 找到卫星轨迹的中间可见点作为标签位置
            visible_points = [p for p in trajectory if p['can_receive']]
            if visible_points:
                # 提取卫星编号（如C06）
                if '(' in sat_name and ')' in sat_name:
                    # 提取括号内的内容
                    sat_id = sat_name[sat_name.find('(')+1:sat_name.find(')')]
                else:
                    # 如果没有括号，使用原始名称的前5个字符
                    sat_id = sat_name[:5]
                
                # 选择中间可见点作为标签位置
                mid_idx = len(visible_points) // 2
                label_point = visible_points[mid_idx]
                label_x = np.deg2rad(label_point['azimuth'])
                label_y = label_point['elevation']
                
                # 添加卫星标签，带背景框
                ax.text(label_x, label_y, sat_id, 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       zorder=10)  # 确保标签在最上层
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    def plot_3d_antenna_view(self, satellite_data, ground_lat=39.9042, ground_lon=116.4074, elevation_threshold=5):
        """
        绘制3D天线视图和2D天空图，包括：
        1. 左侧：3D天线平面、天线坐标系和卫星方向矢量
        2. 右侧：2D天空图，显示卫星在天线本地坐标系中的位置
        3. 姿态角控制滑块（方位角、俯仰角、滚转角），同步更新两个视图
        
        参数:
        satellite_data: 字典列表，每个字典包含卫星名称、方位角(azimuth)和仰角(elevation)
        ground_lat: 地面站纬度（度）
        ground_lon: 地面站经度（度）
        elevation_threshold: 仰角阈值（度）
        """
        # 创建图形和两个子图
        fig = plt.figure(figsize=(18, 10))
        
        # 左侧：3D天线视图
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_3d.view_init(elev=20, azim=30)
        ax_3d.set_xlabel('X (East)')
        ax_3d.set_ylabel('Y (North)')
        ax_3d.set_zlabel('Z (Up)')
        ax_3d.set_title('3D Antenna View with Satellite Direction Vectors')
        
        # 设置3D坐标轴范围
        axis_range = 2.0
        ax_3d.set_xlim([-axis_range, axis_range])
        ax_3d.set_ylim([-axis_range, axis_range])
        ax_3d.set_zlim([-axis_range, axis_range])
        ax_3d.set_aspect('equal')
        
        # 初始姿态角，存储为局部变量，方便重置使用
        initial_az = 0
        initial_el = 0
        initial_roll = 0
        
        # 右侧：2D天空图
        ax_sky = fig.add_subplot(122, projection='polar')
        ax_sky.set_title('Satellite Sky Plot (Local Coordinate System)')
        
        # 1. 绘制当地地平坐标系（固定）
        # East (X)
        ax_3d.quiver(0, 0, 0, 1, 0, 0, color='red', length=1.5, arrow_length_ratio=0.1, label='Local East (X)')
        # North (Y)
        ax_3d.quiver(0, 0, 0, 0, 1, 0, color='green', length=1.5, arrow_length_ratio=0.1, label='Local North (Y)')
        # Up (Z)
        ax_3d.quiver(0, 0, 0, 0, 0, 1, color='blue', length=1.5, arrow_length_ratio=0.1, label='Local Up (Z)')
        
        antenna_vecs = [[1,0,0], [0,1,0], [0,0,1]]
        colors = ['orange', 'cyan', 'magenta']
        labels = ['Antenna X', 'Antenna Y', 'Antenna Z']
        # 存储动态艺术家对象
        dynamic_artists = []
            
        for i, (u, v, w) in enumerate(antenna_vecs):
            q = ax_3d.quiver(0, 0, 0, u, v, w, 
                        color=colors[i], length=1.2, arrow_length_ratio=0.1, 
                           label=labels[i])
            dynamic_artists.append(q)
        #print(dynamic_artists.__len__())
        # 2. 绘制天线平面（矩形平面）
        # 天线平面尺寸
        antenna_size = 1.0
        # 生成天线平面的四个角点（初始姿态）
        plane_corners = np.array([
            [-antenna_size/2, -antenna_size/2, 0],
            [ antenna_size/2, -antenna_size/2, 0],
            [ antenna_size/2,  antenna_size/2, 0],
            [-antenna_size/2,  antenna_size/2, 0],
            [-antenna_size/2, -antenna_size/2, 0]  # 闭合
        ])
        
        # 绘制天线平面
        antenna_plane = ax_3d.plot(plane_corners[:, 0], plane_corners[:, 1], plane_corners[:, 2], 
                              color='gray', linewidth=2, alpha=0.7, label='Antenna Plane')[0]
        
        # 存储卫星名称文本对象
        satellite_texts = []
        
        # 绘制卫星名称（固定位置）
        # 初始可见卫星
        initial_visible_satellites = [sat for sat in satellite_data if sat['elevation'] >= elevation_threshold]
        for sat in initial_visible_satellites:
            # 转换方位角和仰角到方向矢量
            az_rad = np.radians(sat['azimuth'])
            el_rad = np.radians(sat['elevation'])
            
            # 计算单位方向矢量（当地地平坐标系）
            x = np.sin(az_rad) * np.cos(el_rad)
            y = np.cos(az_rad) * np.cos(el_rad)
            z = np.sin(el_rad)
            
            # 标记卫星名称
            text = ax_3d.text(x*2, y*2, z*2, sat['name'], fontsize=8, ha='center', va='center')
            satellite_texts.append(text)
        
        # 4. 旋转矩阵函数（Z-Y-X欧拉角）
        def get_rotation_matrix(az, el, roll):
            """
            计算旋转矩阵（Z-Y-X欧拉角）
            """
            az_rad = np.radians(az)
            el_rad = np.radians(el)
            roll_rad = np.radians(roll)
            
            # 绕Z轴旋转（方位角）
            Rz = np.array([
                [np.cos(az_rad), -np.sin(az_rad), 0],
                [np.sin(az_rad), np.cos(az_rad), 0],
                [0, 0, 1]
            ])
            
            # 绕Y轴旋转（俯仰角）
            Ry = np.array([
                [np.cos(el_rad), 0, np.sin(el_rad)],
                [0, 1, 0],
                [-np.sin(el_rad), 0, np.cos(el_rad)]
            ])
            
            # 绕X轴旋转（滚转角）
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0, np.sin(roll_rad), np.cos(roll_rad)]
            ])
            
            # 组合旋转矩阵（Z-Y-X顺序）
            R = Rz @ Ry @ Rx
            return R
        
        # 5. 更新天空图函数
        def update_sky_plot(az, el, roll):
            """
            更新2D天空图，显示当前姿态下的卫星位置
            """
            # 清除当前天空图
            ax_sky.clear()
            
            # 使用nonlocal关键字访问外部变量cbar
            #nonlocal cbar
            
            # 移除所有不是3D轴、天空图轴和滑块轴的轴
            # 这是一种更安全的方式来移除颜色条
            axes_to_keep = [ax_3d, ax_sky, ax_az, ax_el, ax_roll]
            for cax in fig.axes.copy():
                if cax not in axes_to_keep:
                    try:
                        cax.remove()
                    except (KeyError, ValueError):
                        # 忽略已经被移除的轴
                        pass
            
            # 重置颜色条引用
            cbar = None
            
            # 设置极坐标参数
            ax_sky.set_theta_zero_location('N')  # 0度在正北方向
            ax_sky.set_theta_direction(-1)  # 顺时针方向
            ax_sky.set_rlim(0, 90)  # 径向范围从0到90度（仰角）
            ax_sky.set_rlabel_position(135)  # 半径标签位置
            
            # 反转径向轴，使得90度（天顶）在中心，0度（地平线）在外围
            ax_sky.set_yticks([0, 5, 30, 60, 90])
            ax_sky.set_yticklabels(['0°', '5°','30°', '60°', '90°'])
            ax_sky.set_ylim(90, 0)  # 反转径向轴
            
            # 绘制同心圆（仰角线）
            ax_sky.grid(True, linestyle='--', alpha=0.7)
            
            # 标记方位角方向
            ax_sky.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax_sky.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
            # 当前姿态
            current_attitude = {
                'azimuth': az,
                'elevation': el,
                'roll': roll
            }
            
            # 处理所有卫星，计算本地坐标并重新判断可见性
            processed_satellites = []
            
            for sat in satellite_data:
                global_az = sat['azimuth']
                global_el = sat['elevation']
                # 检查doppler_shift键是否存在，不存在则使用默认值0
                doppler_shift = sat.get('doppler_shift', 0)
                
                # 转换为本地坐标系
                local_az, local_el = self._euler_rotation(
                    global_az, global_el,
                    current_attitude['azimuth'],
                    current_attitude['elevation'],
                    current_attitude['roll']
                )
                
                # 正确的可见性判断逻辑：
                # 1. 全局仰角 > 0°（卫星真的在地平线以上）
                # 2. 并且本地仰角 >= 门限（卫星在天线视野范围内）
                is_visible = (global_el > 0) and (local_el >= elevation_threshold)
                
                processed_satellites.append({
                    'sat': sat,
                    'local_az': local_az,
                    'local_el': local_el,
                    'is_visible': is_visible,
                    'doppler_shift': doppler_shift
                })
            
            # 分离可见和不可见卫星
            visible_satellites = [p for p in processed_satellites if p['is_visible']]
            invisible_satellites = [p for p in processed_satellites if not p['is_visible']]
            
            # 绘制不可见卫星（灰色，较小）
            if invisible_satellites:
                az_invisible = np.deg2rad([p['local_az'] for p in invisible_satellites])
                el_invisible = [p['local_el'] for p in invisible_satellites]
                ax_sky.scatter(az_invisible, el_invisible, s=30, c='gray', marker='o', alpha=0.5, label='不可见卫星')
            
            # 绘制可见卫星（彩色，较大）
            if visible_satellites:
                az_visible = np.deg2rad([p['local_az'] for p in visible_satellites])
                el_visible = [p['local_el'] for p in visible_satellites]
                doppler_visible = [p['doppler_shift'] for p in visible_satellites]
                
                # 绘制卫星，颜色表示多普勒频移
                scatter = ax_sky.scatter(az_visible, el_visible, s=80, c=doppler_visible, cmap='coolwarm', 
                                    marker='o', alpha=0.8, label='可见卫星')
                
                # 添加颜色条，显示多普勒频移
                #cbar = fig.colorbar(scatter, ax=ax_sky, orientation='vertical', pad=0.1)
                #cbar.set_label('多普勒频移 (Hz)')
                
                # 为可见卫星添加标签（限制数量）
                max_labels = 20
                for i, p in enumerate(visible_satellites[:max_labels]):
                    sat = p['sat']
                    az_rad = np.deg2rad(p['local_az'])
                    sat_el = p['local_el']  # 使用不同的变量名，避免与函数参数el冲突
                    
                    # 提取卫星编号
                    sat_name = sat['name']
                    if '(' in sat_name and ')' in sat_name:
                        sat_id = sat_name[sat_name.find('(')+1:sat_name.find(')')]
                    else:
                        sat_id = sat_name[:5]
                    
                    ax_sky.text(az_rad, sat_el, sat_id, ha='center', va='center', fontsize=8)
            
            # 绘制阵列指向方向
            nue_az = [0,0,90]
            nue_el = [0,90,0]
            lnue_az, lnue_el = self._euler_rotation(
                    nue_az, nue_el,
                    current_attitude['azimuth'],
                    current_attitude['elevation'],
                    current_attitude['roll']
                )
            ax_sky.plot(np.deg2rad([0, lnue_az[0]]), [90, lnue_el[0]], color='green', linewidth=1, label='N')
            ax_sky.plot(np.deg2rad([0, lnue_az[1]]), [90, lnue_el[1]], color='blue', linewidth=1, label='U')
            ax_sky.plot(np.deg2rad([0, lnue_az[2]]), [90, lnue_el[2]], color='red', linewidth=1, label='E')
            ax_sky.scatter(np.deg2rad(lnue_az), lnue_el, s=200, color='red', marker='+')
            
            ax_sky.set_title(f'Satellite Sky Plot (Local Coordinate System)\n当前姿态: 方位角={az:.1f}°，俯仰角={el:.1f}°，滚转角={roll:.1f}°')
            ax_sky.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # 6. 绘制函数：用于重绘所有动态元素
        def draw_elements(az, el, roll):
            """
            根据当前姿态角重绘所有动态元素
            """
            # 计算旋转矩阵
            R = get_rotation_matrix(az, el, roll)
            
            # 更新天线平面
            rotated_corners = np.dot(plane_corners, R.T)
            antenna_plane.set_data(rotated_corners[:, 0], rotated_corners[:, 1])
            antenna_plane.set_3d_properties(rotated_corners[:, 2])
            
            # 清除3D子图中的动态元素
            # 清除所有存储的动态艺术家对象
            for artist in dynamic_artists:
                artist.remove()
            dynamic_artists.clear()
            #print("draw elements:",dynamic_artists.__len__())
            # 额外清除所有可能的动态元素，确保没有残留
            # 清除quiver生成的箭头（PolyCollection）
            for quiver_artist in ax_3d.collections:
                if quiver_artist in dynamic_artists:
                    quiver_artist.remove()
            # 清除所有线条
            for line in ax_3d.lines:
                if line in dynamic_artists:
                    line.remove()
            # 清除所有补丁
            for patch in ax_3d.patches:
                if patch in dynamic_artists:
                    patch.remove()
            
            # 绘制天线坐标系（当前姿态）
            antenna_vecs = [[1,0,0], [0,1,0], [0,0,1]]
            colors = ['orange', 'cyan', 'magenta']
            labels = ['Antenna X', 'Antenna Y', 'Antenna Z']
            
            for i, (u, v, w) in enumerate(antenna_vecs):
                # 计算旋转后的向量
                rotated_vec = R @ np.array([u, v, w])
                # 绘制天线坐标轴
                q = ax_3d.quiver(0, 0, 0, rotated_vec[0], rotated_vec[1], rotated_vec[2], 
                            color=colors[i], length=1.2, arrow_length_ratio=0.1, 
                            label=labels[i])
                dynamic_artists.append(q)
            
            # 重新计算可见卫星，基于当前姿态
            current_visible_satellites = [sat for sat in satellite_data if sat['elevation'] >= elevation_threshold]
            
            # 绘制卫星方向矢量
            for sat in current_visible_satellites:
                # 计算卫星方向矢量
                az_rad = np.radians(sat['azimuth'])
                el_rad = np.radians(sat['elevation'])
                x = np.sin(az_rad) * np.cos(el_rad)
                y = np.cos(az_rad) * np.cos(el_rad)
                z = np.sin(el_rad)
                
                # 绘制方向矢量
                q = ax_3d.quiver(0, 0, 0, x, y, z, color='purple', length=1.8, 
                            arrow_length_ratio=0.1, alpha=0.7, 
                            label="Satellite Vectors" if sat == current_visible_satellites[0] else "")
                dynamic_artists.append(q)
            
            # 更新2D天空图
            update_sky_plot(az, el, roll)
        
        
        # 7. 添加姿态角控制滑块
        # 创建滑块区域
        ax_az = plt.axes([0.25, 0.01, 0.65, 0.03])  # [left, bottom, width, height]
        ax_el = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_roll = plt.axes([0.25, 0.09, 0.65, 0.03])
        
        # 创建滑块
        slider_az = Slider(ax_az, '方位角 (°)', -180.0, 180.0, valinit=initial_az, valfmt='%0.1f')
        slider_el = Slider(ax_el, '俯仰角 (°)', -90.0, 90.0, valinit=initial_el, valfmt='%0.1f')
        slider_roll = Slider(ax_roll, '滚转角 (°)', -180.0, 180.0, valinit=initial_roll, valfmt='%0.1f')
        
        # 修复Slider的_format方法，确保使用ASCII负号
        def fixed_format(val):
            # 直接格式化，使用ASCII负号
            return f'{val:.1f}'.replace('−', '-')
        
        # 应用修复到所有滑块
        slider_az._format = fixed_format
        slider_el._format = fixed_format
        slider_roll._format = fixed_format
        
        # 创建一个颜色条引用，用于管理颜色条
        #cbar = None
        
        # 更新函数
        def update(val):
            # 直接从滑块获取当前值，这是最可靠的方式
            az = slider_az.val
            el = slider_el.val
            roll = slider_roll.val
            
            # 输出姿态信息到终端，并立即刷新缓冲区
            print(f"更新姿态: 方位角={az:.1f}, 俯仰角={el:.1f}, 滚转角={roll:.1f}")
            import sys
            sys.stdout.flush()
            
            # 重绘动态元素和天空图
            draw_elements(az, el, roll)
            
            # 明确触发图形重绘
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        # 连接滑块事件
        slider_az.on_changed(update)
        slider_el.on_changed(update)
        slider_roll.on_changed(update)
        
        # 添加键盘快捷键重置功能，按R键重置
        def on_key_press(event):
            if event.key == 'r' or event.key == 'R':
                print("=== 检测到键盘快捷键R，开始重置 ===")
                # 直接设置滑块数值为初始值
                slider_az.set_val(initial_az)
                slider_el.set_val(initial_el)
                slider_roll.set_val(initial_roll)
                # 手动调用update函数更新天线姿态
                update(None)
                print("=== 重置完成 ===")
        
        # 连接键盘事件
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 添加图例
        ax_3d.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # 调整布局
        plt.subplots_adjust(bottom=0.15)
        
        # 初始绘制
        draw_elements(initial_az, initial_el, initial_roll)
    
    def show_all_plots(self):
        """显示所有绘制的图形"""
        plt.show()
