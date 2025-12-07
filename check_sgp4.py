import sgp4
print(f"sgp4版本: {sgp4.__version__}")
print("\n--- sgp4库的顶层属性 ---\n")
print([attr for attr in dir(sgp4) if not attr.startswith('_')])

print("\n--- 尝试导入不同的sgp4子模块 ---\n")

# 尝试导入不同的子模块
submodules = ['api', 'propagation', 'conveniences', 'earth_gravity', 'io', 'errors']
for submodule in submodules:
    try:
        exec(f"import sgp4.{submodule}")
        print(f"✓ sgp4.{submodule} 导入成功")
        print(f"  属性: {[attr for attr in dir(eval(f'sgp4.{submodule}')) if not attr.startswith('_')]}")
    except Exception as e:
        print(f"✗ sgp4.{submodule} 导入失败: {e}")

print("\n--- 尝试直接导入Satrec类 ---\n")
try:
    from sgp4 import Satrec
    print("✓ 成功导入Satrec类")
    print(f"  Satrec属性: {[attr for attr in dir(Satrec) if not attr.startswith('_')][:10]}...")
except Exception as e:
    print(f"✗ 导入Satrec类失败: {e}")

print("\n--- 检查是否有坐标转换相关的函数 ---\n")

# 搜索所有子模块中是否有坐标转换相关的函数
for submodule in submodules:
    try:
        exec(f"import sgp4.{submodule}")
        module = eval(f'sgp4.{submodule}')
        for attr in dir(module):
            if not attr.startswith('_'):
                attr_obj = getattr(module, attr)
                if callable(attr_obj):
                    # 检查函数名是否包含坐标转换相关的关键词
                    if any(keyword in attr.lower() for keyword in ['eci', 'ecef', 'convert', 'transform', 'coord']):
                        print(f"✓ 在sgp4.{submodule}中发现坐标转换相关函数: {attr}")
    except Exception:
        continue

print("\n--- 测试Satrec.sgp4方法返回值 ---\n")
try:
    from sgp4 import Satrec
    # 使用简单的TLE数据测试
    tle_line1 = "1 25544U 98067A   24001.12345678  .00000000  00000-0  00000+0 0  9993"
    tle_line2 = "2 25544  51.6458 123.4567 0000000  0.0000  0.0000 15.43212345  1000"
    
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    print("✓ 成功创建Satrec对象")
    
    # 测试sgp4方法
    import datetime
    from datetime import datetime
    time_now = datetime.now()
    from sgp4 import jday
    jd, fr = jday(time_now.year, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second)
    
    e, r, v = satellite.sgp4(jd, fr)
    print(f"✓ sgp4方法返回值: error={e}, position={r}, velocity={v}")
    print(f"  位置类型: {type(r)}, 速度类型: {type(v)}")
    
except Exception as e:
    print(f"✗ 测试Satrec.sgp4方法失败: {e}")
