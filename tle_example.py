from satellite_signal import SatelliteSignal
from visualization import Visualization
from datetime import datetime, timedelta
import os

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

# 使用下载的北斗卫星TLE文件
tle_file_path = "beidou.tle"
if os.path.exists(tle_file_path):
    print(f"从 {tle_file_path} 读取北斗卫星TLE数据...")
    tle_data = read_tle_file(tle_file_path)
    print(f"成功读取 {len(tle_data)} 颗北斗卫星的TLE数据")
else:
    print(f"警告：{tle_file_path} 文件不存在，将使用默认TLE数据")
    # 默认TLE数据（备用）
    tle_data = {
        "BDS-2 IGSO": {
            "line1": "1 37820U 11060A   24001.12345678  .00000000  00000-0  00000+0 0  9993",
            "line2": "2 37820  55.0000 123.4567 0000000  0.0000  0.0000  1.00270000  1000"
        },
        "BDS-3 MEO": {
            "line1": "1 44404U 19023A   24001.12345678  .00000000  00000-0  00000+0 0  9993",
            "line2": "2 44404  55.0000 123.4567 0000000  0.0000  0.0000  1.00270000  1000"
        },
        "ISS": {
            "line1": "1 25544U 98067A   24001.12345678  .00016717  00000-0  10270-3 0  9053",
            "line2": "2 25544  51.6458 123.4567 0001234  45.6789 134.3210 15.43212345678901"
        }
    }

# 地面站位置示例
# 北京
beijing = {
    "name": "北京",
    "lat": 39.9042,
    "lon": 116.4074,
    "alt": 0.0
}

# 上海
shanghai = {
    "name": "上海",
    "lat": 31.2304,
    "lon": 121.4737,
    "alt": 0.0
}

# 广州
guangzhou = {
    "name": "广州",
    "lat": 23.1291,
    "lon": 113.2644,
    "alt": 0.0
}

# 地面站列表
ground_stations = [beijing, shanghai, guangzhou]

# 创建SatelliteSignal对象
sat_signal = SatelliteSignal(frequency=1.5e9)  # 1.5GHz信号

# 当前时间
time_now = datetime.now()

print("=== 卫星信号参数计算示例 ===")
print(f"当前时间: {time_now}")
print()

# 初始化可视化对象
viz = Visualization()

# 为每个地面站计算每个卫星的信号参数
for station in ground_stations:
    print(f"=== {station['name']}地面站 ===")
    print(f"位置: 纬度 {station['lat']}°N, 经度 {station['lon']}°E, 海拔 {station['alt']}km")
    print()
    
    # 收集该地面站的卫星数据，用于绘制天空图
    station_satellite_data = []
    
    for sat_name, tle in tle_data.items():
        # 计算卫星信号参数
        params = sat_signal.get_satellite_signal_parameters(
            tle['line1'], tle['line2'], 
            station['lat'], station['lon'], station['alt'], 
            time_now, 
            elevation_threshold=5
        )
        
        print(f"卫星: {sat_name}")
        print(f"  方位角: {params['azimuth']:.2f}°")
        print(f"  仰角: {params['elevation']:.2f}°")
        print(f"  多普勒频移: {params['doppler_shift']:.2f}Hz")
        print(f"  是否可以接收: {'是' if params['can_receive'] else '否'}")
        print()
        
        # 收集卫星数据
        station_satellite_data.append({
            'name': sat_name,
            'azimuth': params['azimuth'],
            'elevation': params['elevation'],
            'doppler_shift': params['doppler_shift'],
            'can_receive': params['can_receive']
        })
    
    # 绘制该地面站的天空图
    # max_labels参数控制显示的标签数量，默认10个
    # 设置为20可以显示更多标签，但过多标签会导致重叠影响可读性
    viz.plot_sky_plot(
        station_satellite_data, 
        title=f"{station['name']}地面站可见卫星天空图 ({time_now.strftime('%Y-%m-%d %H:%M:%S')})")

# 读取TLE文件示例（使用已下载的北斗TLE文件）
print("=== TLE文件读取示例 ===")
print(f"从 {tle_file_path} 中读取到 {len(tle_data)} 颗卫星的TLE数据")
print("前5颗卫星：")
for i, sat_name in enumerate(list(tle_data.keys())[:5]):
    print(f"  {i+1}. {sat_name}")
print()

# 演示如何计算卫星可见窗口
def calculate_visibility_window(tle_line1, tle_line2, ground_lat, ground_lon, ground_alt, start_time, duration_hours, elevation_threshold=5):
    """
    计算卫星在一段时间内的可见窗口
    
    参数:
        tle_line1: TLE第一行
        tle_line2: TLE第二行
        ground_lat: 地面站纬度（度）
        ground_lon: 地面站经度（度）
        ground_alt: 地面站海拔（km）
        start_time: 开始时间
        duration_hours: 持续时间（小时）
        elevation_threshold: 仰角门限（度）
    
    返回:
        可见窗口列表，每个元素为(开始时间, 结束时间)
    """
    visibility_windows = []
    current_time = start_time
    end_time = start_time + timedelta(hours=duration_hours)
    
    # 计算步长（秒）
    step_seconds = 60  # 每分钟计算一次
    
    # 标志：当前是否可见
    is_visible = False
    window_start = None
    
    while current_time <= end_time:
        params = sat_signal.get_satellite_signal_parameters(
            tle_line1, tle_line2, ground_lat, ground_lon, ground_alt, current_time, elevation_threshold
        )
        
        if params['can_receive'] and not is_visible:
            # 卫星刚刚可见
            is_visible = True
            window_start = current_time
        elif not params['can_receive'] and is_visible:
            # 卫星刚刚不可见
            is_visible = False
            visibility_windows.append((window_start, current_time))
        
        # 时间递增
        current_time += timedelta(seconds=step_seconds)
    
    # 处理最后一个窗口
    if is_visible:
        visibility_windows.append((window_start, end_time))
    
    return visibility_windows

print("=== 卫星可见窗口计算示例 ===")
# 从北斗卫星中选择一颗进行可见窗口计算
if tle_data:
    # 选择第一颗北斗卫星
    first_sat_name = list(tle_data.keys())[0]
    first_sat_tle = tle_data[first_sat_name]
    
    print(f"计算未来24小时内，北京地面站可见{first_sat_name}的时间窗口：")
    
    # 计算未来24小时的可见窗口
    visibility_windows = calculate_visibility_window(
        first_sat_tle['line1'], first_sat_tle['line2'],
        beijing['lat'], beijing['lon'], beijing['alt'],
        time_now,
        duration_hours=24,
        elevation_threshold=5
    )
    
    if visibility_windows:
        print(f"共找到 {len(visibility_windows)} 个可见窗口：")
        for i, (start, end) in enumerate(visibility_windows, 1):
            duration = end - start
            duration_minutes = duration.total_seconds() / 60
            print(f"  窗口 {i}: {start.strftime('%Y-%m-%d %H:%M:%S')} 至 {end.strftime('%Y-%m-%d %H:%M:%S')}，持续 {duration_minutes:.1f} 分钟")
    else:
        print(f"在未来24小时内，{first_sat_name}在北京地面站不可见")
else:
    print("没有可用的TLE数据来计算可见窗口")

# 演示如何绘制一段时间内的天空图轨迹
print("=== 绘制一段时间内的天空图轨迹 ===")
print("计算未来2小时内的卫星轨迹...")

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

# 选择北京地面站
selected_station = beijing

# 计算未来2小时内的卫星轨迹
trajectories = calculate_satellite_trajectories(
    tle_data, 
    selected_station['lat'], selected_station['lon'], selected_station['alt'],
    time_now, 
    duration_hours=2,  # 2小时
    step_minutes=5  # 每5分钟计算一次
)

# 选择前5颗卫星进行轨迹绘制，避免图形过于复杂
selected_satellites = list(trajectories.keys())[:5]
selected_trajectories = {sat: trajectories[sat] for sat in selected_satellites}

# 绘制轨迹天空图
viz.plot_sky_plot_with_trajectory(
    selected_trajectories,
    title=f"{selected_station['name']}地面站卫星轨迹图（未来2小时）"
)

# 显示所有绘制的图形
viz.show_all_plots()