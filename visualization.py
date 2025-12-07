import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualization:
    def __init__(self):
        """初始化可视化类"""
        # 配置matplotlib参数
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
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
    
    def plot_sky_plot(self, satellite_data, title="Satellite Sky Plot", max_labels=20):
        """
        绘制地面站可见卫星的天空图
        
        参数:
        satellite_data: 字典列表，每个字典包含卫星名称、方位角(azimuth)和仰角(elevation)
        title: str - 图形标题
        max_labels: int - 最多显示的标签数量（默认10）
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
        
        # 分离可见和不可见卫星
        visible_satellites = [sat for sat in satellite_data if sat['can_receive']]
        invisible_satellites = [sat for sat in satellite_data if not sat['can_receive']]
        
        # 绘制不可见卫星（灰色，较小）
        if invisible_satellites:
            az_invisible = np.deg2rad([sat['azimuth'] for sat in invisible_satellites])
            el_invisible = [sat['elevation'] for sat in invisible_satellites]
            ax.scatter(az_invisible, el_invisible, s=30, c='gray', marker='o', alpha=0.5, label='不可见卫星')
        
        # 绘制可见卫星（彩色，较大）
        if visible_satellites:
            az_visible = np.deg2rad([sat['azimuth'] for sat in visible_satellites])
            el_visible = [sat['elevation'] for sat in visible_satellites]
            # 使用多普勒频移作为颜色
            doppler_visible = [sat['doppler_shift'] for sat in visible_satellites]
            
            # 绘制卫星，颜色表示多普勒频移
            scatter = ax.scatter(az_visible, el_visible, s=80, c=doppler_visible, cmap='coolwarm', 
                               marker='o', alpha=0.8, label='可见卫星')
            
            # 添加颜色条，显示多普勒频移
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('多普勒频移 (Hz)')
            
            # 为可见卫星添加标签
            # 默认只显示前max_labels个标签，避免标签重叠影响可读性
            # 可以通过调整max_labels参数来显示更多或更少的标签
            for i, sat in enumerate(visible_satellites[:max_labels]):
                # 提取卫星编号（如C06）
                sat_name = sat['name']
                # 查找括号中的编号，如(C06)
                if '(' in sat_name and ')' in sat_name:
                    # 提取括号内的内容
                    sat_id = sat_name[sat_name.find('(')+1:sat_name.find(')')]
                else:
                    # 如果没有括号，使用原始名称的前5个字符
                    sat_id = sat_name[:5]
                ax.text(np.deg2rad(sat['azimuth']), sat['elevation'], 
                       sat_id, ha='center', va='center', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 如果有更多可见卫星没有显示标签，添加说明
            if len(visible_satellites) > max_labels:
                print(f"提示：共有 {len(visible_satellites)} 颗可见卫星，仅显示前 {max_labels} 个标签以避免重叠。")
                print("可以通过调整 max_labels 参数来显示更多标签。")
        
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
        
        # 为每颗卫星绘制轨迹
        for sat_name, trajectory in satellite_trajectories.items():
            # 只绘制可见弧段的轨迹
            visible_segments = []
            current_segment = []
            
            # 将轨迹分割为可见段和不可见段
            for point in trajectory:
                if point['can_receive']:
                    # 如果当前点可见，添加到当前可见段
                    current_segment.append(point)
                else:
                    # 如果当前点不可见，保存当前可见段并开始新段
                    if current_segment:
                        visible_segments.append(current_segment)
                        current_segment = []
            
            # 保存最后一个可见段
            if current_segment:
                visible_segments.append(current_segment)
            
            # 绘制可见段轨迹
            for i, segment in enumerate(visible_segments):
                # 提取可见段数据
                segment_az = [p['azimuth'] for p in segment]
                segment_el = [p['elevation'] for p in segment]
                segment_az_rad = np.deg2rad(segment_az)
                
                # 绘制可见段轨迹线
                ax.plot(segment_az_rad, segment_el, linewidth=2.0, alpha=0.9, label=f'{sat_name} 轨迹段 {i+1}')
                
                # 绘制可见段轨迹点（绿色）
                ax.scatter(segment_az_rad, segment_el, s=25, c='g', alpha=0.7)
                
                # 标记可见段的起点和终点
                if len(segment) > 1:
                    # 可见段起点（蓝色）
                    ax.scatter([segment_az_rad[0]], [segment_el[0]], s=50, c='blue', marker='o', label=f'{sat_name} 可见段 {i+1} 起点')
                    # 可见段终点（红色）
                    ax.scatter([segment_az_rad[-1]], [segment_el[-1]], s=50, c='red', marker='o', label=f'{sat_name} 可见段 {i+1} 终点')
                    
                    # 在轨迹中间位置添加卫星标签
                    # 提取卫星编号（如C06）
                    if '(' in sat_name and ')' in sat_name:
                        # 提取括号内的内容
                        sat_id = sat_name[sat_name.find('(')+1:sat_name.find(')')]
                    else:
                        # 如果没有括号，使用原始名称的前5个字符
                        sat_id = sat_name[:5]
                    
                    # 选择轨迹中间点作为标签位置
                    mid_idx = len(segment_az_rad) // 2
                    label_x = segment_az_rad[mid_idx]
                    label_y = segment_el[mid_idx]
                    
                    # 添加卫星标签，带背景框
                    ax.text(label_x, label_y, sat_id, 
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           zorder=10)  # 确保标签在最上层
            
            # 只有当没有可见段时，标记整个轨迹的起点和终点
            if not visible_segments and trajectory:
                # 转换方位角为弧度
                az_rad = np.deg2rad([point['azimuth'] for point in trajectory])
                elevations = [point['elevation'] for point in trajectory]
                
                # 标记轨迹起点（蓝色）和终点（红色）
                ax.scatter([az_rad[0]], [elevations[0]], s=50, c='blue', marker='o', label=f'{sat_name} 起点')
                ax.scatter([az_rad[-1]], [elevations[-1]], s=50, c='red', marker='o', label=f'{sat_name} 终点')
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    def show_all_plots(self):
        """显示所有绘制的图形"""
        plt.show()