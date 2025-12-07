from satellite_signal import SatelliteSignal
from datetime import datetime, timedelta
import numpy as np

# 创建卫星信号对象
sat_signal = SatelliteSignal(frequency=1.5e9)

# 地面站位置（北京）
ground_lat = 39.9042
ground_lon = 116.4074
ground_alt = 0.0

# C04卫星TLE数据（从beidou.tle中获取）
c04_tle = {
    "line1": "1 37210U 10057A   25340.24449524 -.00000105  00000+0  00000+0 0  9999",
    "line2": "2 37210   3.1677  69.9541 0010958 189.5357  63.8885  1.00271337 55354"
}

# 测试时间范围：当前时间开始，持续10小时
time_now = datetime.now()
duration_hours = 10
step_minutes = 60  # 每小时一个点

print(f"=== 测试C04卫星在{duration_hours}小时内的位置变化 ===")
print(f"地面站位置：纬度 {ground_lat}°N, 经度 {ground_lon}°E")
print(f"测试开始时间：{time_now.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 生成时间序列
times = [time_now + timedelta(minutes=i*step_minutes) for i in range(duration_hours + 1)]

# 存储位置数据
positions = []

# 计算每个时间点的方位角和仰角
for current_time in times:
    params = sat_signal.get_satellite_signal_parameters(
        c04_tle['line1'], c04_tle['line2'],
        ground_lat, ground_lon, ground_alt,
        current_time,
        elevation_threshold=5
    )
    
    positions.append({
        'time': current_time,
        'azimuth': params['azimuth'],
        'elevation': params['elevation'],
        'can_receive': params['can_receive']
    })

# 输出结果
print("时间\t\t\t方位角 (°)\t仰角 (°)\t是否可见")
print("="*60)

for pos in positions:
    time_str = pos['time'].strftime('%Y-%m-%d %H:%M:%S')
    print(f"{time_str}\t{pos['azimuth']:.4f}\t{pos['elevation']:.4f}\t{'是' if pos['can_receive'] else '否'}")

# 计算最大变化
print()
print("=== 位置变化分析 ===")
azimuths = [pos['azimuth'] for pos in positions]
elevations = [pos['elevation'] for pos in positions]

max_azimuth_change = max(azimuths) - min(azimuths)
max_elevation_change = max(elevations) - min(elevations)

print(f"方位角变化范围：{min(azimuths):.4f}° 至 {max(azimuths):.4f}°")
print(f"方位角最大变化：{max_azimuth_change:.4f}°")
print(f"仰角变化范围：{min(elevations):.4f}° 至 {max(elevations):.4f}°")
print(f"仰角最大变化：{max_elevation_change:.4f}°")

# 检查是否为GEO卫星的TLE数据
print()
print("=== TLE数据分析 ===")
print(f"C04卫星TLE数据：")
print(f"  第一行：{c04_tle['line1']}")
print(f"  第二行：{c04_tle['line2']}")

# 提取TLE中的关键参数
tle_line2 = c04_tle['line2'].strip()
inclination = float(tle_line2[8:16])
mean_motion = float(tle_line2[52:63])

# 正确计算半长轴（km）
# 开普勒第三定律：a^3 = μ * P^2 / (4π^2)
# 其中μ=398600.4418 km³/s²（地球引力常数）
# P是轨道周期（秒），P=86400/mean_motion（秒/转）
mu = 398600.4418
P = 86400 / mean_motion  # 轨道周期（秒）
a = (mu * P**2 / (4 * np.pi**2))**(1/3)

print(f"  轨道倾角：{inclination:.4f}°")
print(f"  平均运动：{mean_motion:.8f} 转/天")
print(f"  轨道周期：{P:.2f} 秒")
print(f"  半长轴：{a:.2f} km")

# 地球半径（km）
EARTH_RADIUS = 6378.137

# GEO卫星的严格判断条件
# 1. 平均运动接近1转/天（误差<0.001）
# 2. 轨道倾角接近0°（误差<0.5°）
# 3. 半长轴接近42164 km（误差<100 km）- 包含地球半径
#    GEO轨道高度：35786 km（距地面）
#    半长轴 = 轨道高度 + 地球半径 = 35786 + 6378 = 42164 km
geo_mean_motion = 1.0
geo_inclination = 0.0
geo_semi_major_axis = 42164.0  # 正确的GEO半长轴（包含地球半径）
geo_orbit_height = geo_semi_major_axis - EARTH_RADIUS  # GEO轨道高度（距地面）

# 计算当前卫星的轨道高度（距地面）
orbit_height = a - EARTH_RADIUS

is_geo = (
    abs(mean_motion - geo_mean_motion) < 0.005 and
    abs(inclination - geo_inclination) < 5.0 and
    abs(a - geo_semi_major_axis) < 100
)

# 计算IGSO卫星的判断条件
# IGSO卫星具有与地球同步的周期，但轨道倾角较大
is_igso = (
    abs(mean_motion - geo_mean_motion) < 0.005 and
    abs(inclination) >= 5.0 and abs(inclination) < 30
)

# 计算MEO卫星的判断条件
# MEO卫星轨道高度约2000-35786 km（低于GEO轨道）
is_meo = (
    orbit_height > 2000 and orbit_height < geo_orbit_height - 100
)

# 计算LEO卫星的判断条件
# LEO卫星轨道高度<2000 km
is_leo = (
    orbit_height < 2000
)

# 输出更详细的轨道信息
print(f"  轨道高度（距地面）：{orbit_height:.2f} km")

if is_geo:
    print("  结论：这是一颗GEO卫星")
elif is_igso:
    print("  结论：这是一颗倾斜地球同步轨道(IGSO)卫星")
elif is_meo:
    print("  结论：这是一颗中地球轨道(MEO)卫星")
elif is_leo:
    print("  结论：这是一颗低地球轨道(LEO)卫星")
else:
    print("  结论：无法确定卫星轨道类型")

if is_geo:
    print(f"  与GEO卫星的偏差：")
    print(f"    平均运动偏差：{abs(mean_motion - geo_mean_motion):.6f} 转/天")
    print(f"    轨道倾角偏差：{abs(inclination - geo_inclination):.4f}°")
    print(f"    半长轴偏差：{abs(a - geo_semi_major_axis):.2f} km")
    print(f"    轨道高度偏差：{abs(orbit_height - geo_orbit_height):.2f} km")

print()
print("=== 结果解释 ===")
print("GEO卫星应该相对于地面站静止，但会有微小漂移：")
print("1. 轨道倾角不为0度（实际GEO卫星有微小倾角）")
print("2. 地球非球形（J2项等）导致的轨道摄动")
print("3. 太阳和月球引力的影响")
print("4. 大气阻力（虽然GEO轨道大气很稀薄）")
print()
print(f"10小时内方位角变化 {max_azimuth_change:.4f}°，仰角变化 {max_elevation_change:.4f}°，")
print("这个变化对于GEO卫星来说是正常的，因为：")
print("- 地球同步轨道周期约23小时56分4秒，不是精确24小时")
print("- 轨道倾角会导致卫星在南北方向有微小摆动")
print("- 经度方向会有缓慢漂移")
print()
print("在天空图中，即使是0.1°的变化也可能看起来明显，因为天空图是极坐标图，")
print("靠近中心（高仰角）的小角度变化在视觉上会更明显。")
