from satellite_signal import SatelliteSignal
from datetime import datetime

# 示例TLE数据 - 国际空间站（ISS）
tle_line1 = "1 25544U 98067A   24001.12345678  .00016717  00000-0  10270-3 0  9053"
tle_line2 = "2 25544  51.6458 123.4567 0001234  45.6789 134.3210 15.43212345678901"

# 地面站位置 - 北京（示例）
ground_lat = 39.9042  # 纬度（度）
ground_lon = 116.4074  # 经度（度）
ground_alt = 0.0  # 海拔（km）

# 当前时间
time = datetime.now()

# 创建SatelliteSignal对象
sat_signal = SatelliteSignal(frequency=1.5e9)  # 1.5GHz信号

# 测试功能
print("=== 卫星信号参数测试 ===")
print(f"时间: {time}")
print(f"地面站位置: 纬度 {ground_lat}°N, 经度 {ground_lon}°E, 海拔 {ground_alt}km")
print(f"TLE数据:")
print(f"  {tle_line1}")
print(f"  {tle_line2}")
print()

# 计算卫星信号参数
params = sat_signal.get_satellite_signal_parameters(
    tle_line1, tle_line2, ground_lat, ground_lon, ground_alt, time, elevation_threshold=5
)

print("=== 计算结果 ===")
print(f"方位角: {params['azimuth']:.2f}°")
print(f"仰角: {params['elevation']:.2f}°")
print(f"多普勒频移: {params['doppler_shift']:.2f}Hz")
print(f"是否可以接收: {'是' if params['can_receive'] else '否'}")
print(f"仰角门限: 5°")

# 测试不同仰角门限
print()
print("=== 不同仰角门限测试 ===")
for threshold in [0, 5, 10, 15]:
    params = sat_signal.get_satellite_signal_parameters(
        tle_line1, tle_line2, ground_lat, ground_lon, ground_alt, time, elevation_threshold=threshold
    )
    print(f"仰角门限 {threshold}°: 仰角 {params['elevation']:.2f}°, 可接收: {'是' if params['can_receive'] else '否'}")

# 测试不同时间点
print()
print("=== 不同时间点测试 ===")
from datetime import timedelta

for minutes in [0, 10, 20, 30]:
    test_time = time + timedelta(minutes=minutes)
    params = sat_signal.get_satellite_signal_parameters(
        tle_line1, tle_line2, ground_lat, ground_lon, ground_alt, test_time, elevation_threshold=5
    )
    print(f"时间 +{minutes}分钟: 方位角 {params['azimuth']:.2f}°, 仰角 {params['elevation']:.2f}°, 多普勒频移 {params['doppler_shift']:.2f}Hz, 可接收: {'是' if params['can_receive'] else '否'}")
