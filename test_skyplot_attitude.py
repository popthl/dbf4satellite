from satellite_signal import SatelliteSignal
from visualization import Visualization
from datetime import datetime
import os


def read_tle_file(file_path):
    """
    从文件中读取TLE数据
    
    参数:
        file_path: TLE文件路径
    
    返回:
        字典，键为卫星名称，值为包含line1和line2的字典
    """
    tle_dict = {}
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            sat_name = lines[i]
            line1 = lines[i+1]
            line2 = lines[i+2]
            tle_dict[sat_name] = {
                'line1': line1,
                'line2': line2
            }
    
    return tle_dict


def main():
    # 1. 初始化参数
    print("初始化测试参数...")
    
    # 使用真实北斗卫星频率（接近B1频段）
    frequency = 1.5e9  # 1.5 GHz
    
    # 地面站位置（北京）
    ground_lat = 39.9042
    ground_lon = 116.4074
    ground_alt = 0.0
    
    # 当前时间
    current_time = datetime.now()
    
    # 北斗TLE文件路径
    tle_file_path = "beidou.tle"
    
    # 从文件中读取TLE数据
    if os.path.exists(tle_file_path):
        print(f"从 {tle_file_path} 读取北斗卫星TLE数据...")
        tle_data = read_tle_file(tle_file_path)
        print(f"成功读取 {len(tle_data)} 颗北斗卫星的TLE数据")
    else:
        print(f"错误：{tle_file_path} 文件不存在")
        exit(1)
    
    # 创建卫星信号生成器，用于计算卫星参数
    signal_generator = SatelliteSignal(frequency=frequency)
    
    # 2. 计算所有卫星的位置参数
    print("\n计算所有卫星的位置参数...")
    all_satellite_data = []
    elevation_threshold = 5  # 仰角阈值5度
    
    for sat_name, tle in tle_data.items():
        # 计算卫星信号参数
        params = signal_generator.get_satellite_signal_parameters(
            tle['line1'], tle['line2'],
            ground_lat, ground_lon, ground_alt,
            current_time,
            elevation_threshold=elevation_threshold
        )
        all_satellite_data.append({
            'name': sat_name,
            'azimuth': params['azimuth'],
            'elevation': params['elevation'],
            'doppler_shift': params['doppler_shift'],
            'can_receive': params['can_receive']
        })
    
    # 统计可见卫星数量
    visible_count = sum(1 for sat in all_satellite_data if sat['can_receive'])
    print(f"可见卫星数量：{visible_count} 颗")
    
    # 3. 创建可视化对象
    viz = Visualization()
    
    # 4. 定义天线姿态参数
    array_attitude = {
        'azimuth': 60,    # 方位角15度
        'elevation': 10,   # 俯仰角10度
        'roll': 5         # 滚转角5度
    }
    
    # 5. 绘制不考虑天线姿态的天空图
    print("\n绘制不考虑天线姿态的天空图...")
    viz.plot_sky_plot(
        all_satellite_data,
        title='北斗卫星天空图（不考虑天线姿态）',
        max_labels=40,
        array_attitude=None  # 不考虑姿态
    )
    
    # 6. 绘制考虑天线姿态的天空图
    print("绘制考虑天线姿态的天空图...")
    viz.plot_sky_plot(
        all_satellite_data,
        title=f'北斗卫星天空图（考虑天线姿态：方位角{array_attitude["azimuth"]}°，俯仰角{array_attitude["elevation"]}°，滚转角{array_attitude["roll"]}°）',
        max_labels=40,
        array_attitude=array_attitude  # 考虑姿态
    )
    
    # 7. 绘制3D天线视图（使用简化的可见卫星列表）
    print("\n绘制3D天线视图...")
    print(f"卫星数据数量: {len(all_satellite_data)}")
    
    # 过滤可见卫星，只使用可见的卫星
    visible_satellites = [sat for sat in all_satellite_data if sat['can_receive']]
    print(f"可见卫星数量: {len(visible_satellites)}")
    
    # 确保有可见卫星才绘制3D视图
    if visible_satellites:
        # 使用3D天线视图
        viz.plot_3d_antenna_view(visible_satellites[:5])  # 只使用前5颗卫星，避免图形过于复杂
    else:
        print("没有可见卫星，跳过3D天线视图绘制")
    
    # 8. 显示所有图形
    print("\n显示所有图形...")
    viz.show_all_plots()
    
    print("\n测试完成！")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")
        print("完整堆栈跟踪:")
        traceback.print_exc()
