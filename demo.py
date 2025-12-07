import numpy as np
from antenna_array import AntennaArray2D
from beamforming import Beamformer
from satellite_signal import SatelliteSignal
from visualization import Visualization

def main():
    # 1. 初始化参数
    print("初始化2D多波束天线阵列...")
    
    # 天线阵列参数
    num_antennas_x = 4
    num_antennas_y = 4
    spacing_x = 0.5  # 波长倍数
    spacing_y = 0.5  # 波长倍数
    frequency = 1e6  # 1 MHz
    
    # 信号参数
    bandwidth = 10e6  # 10 MHz
    sampling_rate = 20e6  # 20 MHz
    duration = 0.001  # 0.1秒
    snr_db = 20  # 20 dB
    
    # 定义三个不同方向，每个方向带不同的多普勒频移
    # 格式：(方位角, 俯仰角, 多普勒频移)
    directions_with_doppler = [
        (30, 45, 10000),   # 方向1：方位角30度，俯仰角45度，多普勒频移1000Hz
        (-20, 30, -5000),  # 方向2：方位角-20度，俯仰角30度，多普勒频移-500Hz
        (60, 60, 20000)    # 方向3：方位角60度，俯仰角60度，多普勒频移2000Hz
    ]
    
    # 2. 创建天线阵列
    antenna_array = AntennaArray2D(
        num_antennas_x=num_antennas_x,
        num_antennas_y=num_antennas_y,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        frequency=frequency
    )
    
    # 3. 创建波束形成器
    beamformer = Beamformer(antenna_array)
    
    # 4. 创建卫星信号生成器
    signal_generator = SatelliteSignal(
        frequency=frequency,
        bandwidth=bandwidth,
        sampling_rate=sampling_rate
    )
    
    # 5. 创建可视化对象
    viz = Visualization()
    
    # 6. 显示阵列参数
    print("\n阵列参数：")
    array_params = antenna_array.get_array_parameters()
    for key, value in array_params.items():
        print(f"{key}: {value}")
    
    # 7. 绘制天线阵列几何布局
    viz.plot_array_geometry(antenna_array.get_antenna_positions())
    
    # 8. Calculate and plot single beam pattern
    print("\nCalculating single beam pattern...")
    az_range = np.linspace(-180, 180, 361)  # Azimuth range (full 360°)
    el_range = np.linspace(0, 90, 91)  # Elevation range
    
    targetn=2
    print(f'指向目标{targetn}的方向：{directions_with_doppler[targetn][0]}°, {directions_with_doppler[targetn][1]}°')
    # 计算指向第一个目标的波束权重
    weights = beamformer.get_beam_weights(directions_with_doppler[targetn][0], directions_with_doppler[targetn][1])
    
    # 1. Beam pattern without attitude adjustment
    # Calculate beam pattern
    pattern_dB = beamformer.calculate_beam_pattern(az_range, el_range, weights)
    
    # Plot hemisphere beam pattern only
    viz.plot_beam_pattern_hemisphere(az_range, el_range, pattern_dB, 
                                    title=f'Hemisphere Beam Pattern (No Attitude) - Az: {directions_with_doppler[0][0]}°, El: {directions_with_doppler[0][1]}°')
    
    # 2. Set array attitude: Azimuth 15°, Elevation 10°, Roll 5°
    print("\nSetting array attitude: Azimuth 15°, Elevation 10°, Roll 5°")
    antenna_array.set_array_attitude(azimuth=15, elevation=10, roll=5)
    # 重新计算权重
    weights_rot = beamformer.get_beam_weights(directions_with_doppler[targetn][0], directions_with_doppler[targetn][1])
    pattern_dB_rot = beamformer.calculate_beam_pattern(az_range, el_range, weights_rot)
    
    # Plot hemisphere beam pattern only
    viz.plot_beam_pattern_hemisphere(az_range, el_range, pattern_dB_rot, 
                                    title=f'Hemisphere Beam Pattern (Attitude: 15°Az, 10°El, 5°Roll) - Az: {directions_with_doppler[0][0]}°, El: {directions_with_doppler[0][1]}°')    
    
    # 3. Set array attitude: Azimuth -30°, Elevation 20°, Roll -10°
    #print("\nSetting array attitude: Azimuth -30°, Elevation 20°, Roll -10°")
    #antenna_array.set_array_attitude(azimuth=-30, elevation=20, roll=-10)
    # 重新计算权重
    #weights_rot2 = beamformer.get_beam_weights(directions_with_doppler[targetn][0], directions_with_doppler[targetn][1])
    #pattern_dB_rot2 = beamformer.calculate_beam_pattern(az_range, el_range, weights_rot2)
    
    # Plot hemisphere beam pattern only
    #viz.plot_beam_pattern_hemisphere(az_range, el_range, pattern_dB_rot2, 
    #                                title=f'Hemisphere Beam Pattern (Attitude: -30°Az, 20°El, -10°Roll) - Az: {directions_with_doppler[0][0]}°, El: {directions_with_doppler[0][1]}°')
    
    # 9. Calculate and plot multi-beam pattern
    print("\nCalculating multi-beam pattern...")
    
    # 计算所有目标方向的波束权重
    multi_weights = beamformer.get_multiple_beam_weights(directions_with_doppler)
    
    # 计算合成波束方向图（所有波束的叠加）
    multi_beam_pattern = np.zeros((len(el_range), len(az_range)))
    
    for i in range(len(directions_with_doppler)):
        pattern = beamformer.calculate_beam_pattern(az_range, el_range, multi_weights[i])
        multi_beam_pattern += 10**(pattern/10)  # 转换为线性单位相加
    
    # 转换回dB
    multi_beam_pattern = 10 * np.log10(np.maximum(multi_beam_pattern, 1e-10))
    multi_beam_pattern -= np.max(multi_beam_pattern)  # 归一化到0 dB
    
    # Plot hemisphere multi-beam pattern only
    viz.plot_beam_pattern_hemisphere(az_range, el_range, multi_beam_pattern, 
                                    title='Multi-Beam Pattern (3 Targets) - Hemisphere')
    
    # Polar beam pattern plotting is disabled
    # print("\nPlotting polar beam pattern...")
    # # 选择中间俯仰角索引
    # el_index = len(el_range) // 2
    # selected_el = el_range[el_index]
    # pattern_slice = pattern_dB[el_index, :]
    # 
    # # 绘制极坐标方向图
    # viz.plot_beam_pattern_polar(az_range, pattern_slice, 
    #                           title=f'Polar Beam Pattern (El: {selected_el}°)')
    
    # 10. 模拟接收卫星信号（三个不同方向，带多普勒频移）
    print("\n模拟接收卫星信号...")
    
    print(f"天线姿态：方位角 {antenna_array.array_azimuth}°，俯仰角 {antenna_array.array_elevation}°，滚转角 {antenna_array.array_roll}°")
    
    # 生成并接收信号（三个方向的信号合成）
    received_signals = signal_generator.receive_signal(
        antenna_array, beamformer, directions_with_doppler, duration, snr_db
    )
    
    print(f"Received signal shape: {received_signals.shape}")
    
    # 11. Apply beamforming processing
    print("\nApplying beamforming processing...")
    
    # 单波束处理
    processed_signal = signal_generator.process_received_signal(received_signals, weights_rot)
    print(f"Single beam processed signal shape: {processed_signal.shape}")
    
    # 多波束处理
    multi_processed_signal = signal_generator.process_received_signal(received_signals, multi_weights)
    print(f"Multi-beam processed signal shape: {multi_processed_signal.shape}")
    
    # 12. Plot received signal comparison
    print("\nPlotting received signal comparison...")
    
    # 获取单个阵元的信号（例如第一个阵元）
    single_element_signal = received_signals[0, :]
    
    # 绘制单个阵元信号和波束形成后信号的对比（时域）
    viz.plot_signal_comparison(single_element_signal, processed_signal, sampling_rate, 
                              title='Signal Comparison (3 Directions with Doppler)')
    
    # 绘制单个阵元信号和波束形成后信号的频谱对比
    viz.plot_signal_spectrum_comparison(single_element_signal, processed_signal, sampling_rate, 
                                      title='Signal Spectrum Comparison (3 Directions with Doppler)')
    
    # 绘制单个阵元信号和多波束形成后信号的对比（时域）
    viz.plot_signal_comparison(single_element_signal, np.sum(multi_processed_signal, axis=0), sampling_rate, 
                              title='Signal Comparison (3 Directions with Doppler) - Multi-Beam')

    # 绘制单个阵元信号和多波束形成后信号的频谱对比
    viz.plot_signal_spectrum_comparison(single_element_signal, np.sum(multi_processed_signal, axis=0), sampling_rate, 
                                      title='Signal Spectrum Comparison (3 Directions with Doppler) - Multi-Beam')

    # 绘制波束形成后的时域信号
    #viz.plot_signal_time_domain(processed_signal, sampling_rate, 
    #                          title='Beamformed Signal (3 Directions with Doppler)')
    
    # 绘制波束形成后的频域信号
    #viz.plot_signal_frequency_domain(processed_signal, sampling_rate, 
    #                               title='Beamformed Signal Spectrum (3 Directions with Doppler)')
    
    
    # 13. Plot all elements time domain signal (first 2μs)
    #print("\nPlotting all elements time domain signal (first 2μs)...")
    #viz.plot_all_elements_time_domain(received_signals, sampling_rate, time_range_us=1000, 
    #                                 title="All Elements Time Domain Signal (First 2μs)")
    
    # 绘制所有阵元的时域信号（分开显示）
    #viz.plot_all_elements_time_domain_separate(received_signals, sampling_rate, time_range_us=2, 
    #                                         title="All Elements Time Domain Signal (Separate, First 2μs)")
    
    # 14. Plot element phase differences
    #print("\nPlotting element phase differences...")
    #viz.plot_element_phase_differences(received_signals, reference_element=0, 
    #                                 title="Element Phase Differences (Relative to Element 0)")
    
    # 绘制2D相位分布
    #viz.plot_phase_distribution_2d(received_signals, num_antennas_x=8, num_antennas_y=8, 
    #                              reference_element=0, 
    #                              title="2D Phase Distribution (Relative to Element 0)")
    
    # 13. Display all plots
    print("\nDisplaying all plots...")
    viz.show_all_plots()
    
    print("\n2D Multi-beam antenna array demo completed!")

if __name__ == "__main__":
    main()
