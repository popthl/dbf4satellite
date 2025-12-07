import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 查找所有可用字体
print("查找系统中可用的中文字体...")
all_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
chinese_fonts = []

# 筛选中文字体
for font_path in all_fonts:
    try:
        font = fm.FontProperties(fname=font_path)
        # 测试中文字符是否能正常显示
        if font.get_name() and any(ord(c) > 127 for c in font.get_name()):
            chinese_fonts.append((font.get_name(), font_path))
    except Exception:
        continue

# 打印找到的中文字体
print(f"找到 {len(chinese_fonts)} 种中文字体：")
for i, (font_name, font_path) in enumerate(chinese_fonts):
    print(f"{i+1}. {font_name} - {font_path}")

# 测试并显示字体效果
if chinese_fonts:
    print("\n测试字体显示效果：")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (font_name, font_path) in enumerate(chinese_fonts[:5]):  # 测试前5种字体
        ax.text(0.1, 0.9 - i*0.15, f"测试中文显示：{font_name}", 
                fontproperties=fm.FontProperties(fname=font_path, size=12))
    ax.set_title("中文字体测试", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("chinese_font_test.png", dpi=100)
    print("字体测试结果已保存为 chinese_font_test.png")
else:
    print("未找到可用的中文字体。")
    # 打印所有可用字体名称
    print("\n系统中所有可用字体：")
    for font in fm.fontManager.ttflist[:20]:  # 只显示前20种
        print(font.name)
