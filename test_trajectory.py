from satellite_signal import SatelliteSignal
from visualization import Visualization
from datetime import datetime, timedelta
import os

# 使用下载的北斗卫星TLE文件
tle_file_path = "beidou.tle"

# 从文件中读取TLE数据的函数
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

# 读取TLE数据
if os.path.exists(tle_file_path):
    print(f"从 {tle_file_path} 读取北斗卫星TLE数据...")
    tle_data = read_tle_file(tle_file_path)
    print(f"成功读取 {len(tle_data)} 颗北斗卫星的TLE数据")
else:
    print(f"错误：{tle_file_path} 文件不存在")
    exit(1)

# 创建SatelliteSignal对象
sat_signal = SatelliteSignal(frequency=1.5e9)  # 1.5GHz信号

# 当前时间
time_now = datetime.now()

# 初始化可视化对象
viz = Visualization()

# 计算一段时间内的卫星轨迹
def calculate_satellite_trajectories(tle_data, ground_lat, ground_lon, ground_alt, start_time, duration_hours, step_minutes=5):
    """
    计算一段时间内的卫星轨迹
    
    参数:
    tle_data: 字典，键为卫星名称，值为包含line1和line2的字典
    ground_lat: 地面站纬度（度）
    ground_lon: 地面站经度（度）
    ground_alt: 地面站海拔（km）
    start_time: 开始时间
    duration_hours: 持续时间（小时）
    step_minutes: 计算步长（分钟）
    
    返回:
    字典，键为卫星名称，值为包含时间、方位角、仰角、多普勒频移和接收状态的列表
    """
    trajectories = {}
    step_seconds = step_minutes * 60
    end_time = start_time + timedelta(hours=duration_hours)
    
    # 生成时间序列
    times = [start_time + timedelta(seconds=i*step_seconds) for i in range(int(duration_hours*3600/step_seconds) + 1)]
    
    for sat_name, tle in tle_data.items():
        trajectories[sat_name] = []
        for current_time in times:
            # 计算卫星信号参数
            params = sat_signal.get_satellite_signal_parameters(
                tle['line1'], tle['line2'],
                ground_lat, ground_lon, ground_alt,
                current_time,
                elevation_threshold=5
            )
            
            # 保存轨迹点
            trajectories[sat_name].append({
                'time': current_time,
                'azimuth': params['azimuth'],
                'elevation': params['elevation'],
                'doppler_shift': params['doppler_shift'],
                'can_receive': params['can_receive']
            })
    
    return trajectories

# 北京地面站
beijing = {
    "name": "北京",
    "lat": 39.9042,
    "lon": 116.4074,
    "alt": 0.0
}

print("=== 计算卫星轨迹 ===")
print("计算未来1小时内的卫星轨迹...")

# 计算未来1小时内的卫星轨迹，每5分钟一个点
trajectories = calculate_satellite_trajectories(
    tle_data, 
    beijing['lat'], beijing['lon'], beijing['alt'],
    time_now, 
    duration_hours=10,  # 1小时
    step_minutes=5  # 每5分钟计算一次
)

print(f"成功计算 {len(trajectories)} 颗卫星的轨迹")
print("每颗卫星有 {len(next(iter(trajectories.values())))} 个轨迹点")

# 选择前3颗卫星进行轨迹绘制，避免图形过于复杂
selected_satellites = list(trajectories.keys())[:3]
selected_trajectories = {sat: trajectories[sat] for sat in selected_satellites}
print(f"选择 {len(selected_trajectories)} 颗卫星进行轨迹绘制")

print("=== 绘制轨迹天空图 ===")

# 绘制轨迹天空图
viz.plot_sky_plot_with_trajectory(
    selected_trajectories,
    title=f"{beijing['name']}地面站卫星轨迹图（未来1小时）"
)

print("轨迹天空图绘制完成，正在显示...")

# 显示所有绘制的图形
viz.show_all_plots()

print("程序结束")