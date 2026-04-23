import requests
import json
import os
import time

# --- 配置区 ---
API_URL = "http://10.2.0.84:8049/tts/generate"
VOICE_ID = "jiayan"
OUTPUT_DIR = "test_results"
# 测试数据
test_items = [
    # --- [第一组] 10-15字区间 (共30条，重点观察启动开销与短句RTF) ---
    {"uttid": "0", "syn_text": "你好，很高兴见到你。"},
    {"uttid": "1", "syn_text": "今天的阳光真的很明媚。"},
    {"uttid": "2", "syn_text": "正在测试接口的稳定性。"},
    {"uttid": "3", "syn_text": "这只猫的名字叫作丁满。"},
    {"uttid": "4", "syn_text": "生活要保持积极和乐观。"},
    {"uttid": "5", "syn_text": "这里的空气感觉非常清新。"},
    {"uttid": "6", "syn_text": "保持专注才能提高效率。"},
    {"uttid": "7", "syn_text": "记得每天都要按时吃早饭。"},
    {"uttid": "8", "syn_text": "晚风吹着感觉非常凉快。"},
    {"uttid": "9", "syn_text": "静态图录制已经成功开启。"},
    {"uttid": "10", "syn_text": "不断提高模型推理的能力。"},
    {"uttid": "11", "syn_text": "非常欢迎你来到智能世界。"},
    {"uttid": "12", "syn_text": "这里的风景确实特别漂亮。"},
    {"uttid": "13", "syn_text": "我们需要不断地学习进步。"},
    {"uttid": "14", "syn_text": "这个苹果的味道非常清甜。"},
    {"uttid": "15", "syn_text": "这种感觉真的非常奇妙。"},
    {"uttid": "16", "syn_text": "衷心祝愿你今天过得愉快。"},
    {"uttid": "17", "syn_text": "建筑风格具有鲜明的特色。"},
    {"uttid": "18", "syn_text": "出门记得带好自己的雨伞。"},
    {"uttid": "19", "syn_text": "合成声音听起来非常真实。"},
    {"uttid": "20", "syn_text": "这种设定确实是非常巧妙。"},
    {"uttid": "21", "syn_text": "目前已经完成了初步测试。"},
    {"uttid": "22", "syn_text": "请确认您的收货地址信息。"},
    {"uttid": "23", "syn_text": "这是一个充满希望的新起点。"},
    {"uttid": "24", "syn_text": "请保持手机通讯信号通畅。"},
    {"uttid": "25", "syn_text": "这里的服务态度非常专业。"},
    {"uttid": "26", "syn_text": "每一份努力都会有回报的。"},
    {"uttid": "27", "syn_text": "这项技术的发展速度极快。"},
    {"uttid": "28", "syn_text": "我非常喜欢这首歌的旋律。"},
    {"uttid": "29", "syn_text": "这个项目的进展非常顺利。"},

    # --- [第二组] 15-20字区间 (共15条，观察RTF下降趋势) ---
    {"uttid": "30", "syn_text": "这里的优美风景让我感到心情非常放松。"},
    {"uttid": "31", "syn_text": "弟弟正在学校的操场上开心地踢着足球。"},
    {"uttid": "32", "syn_text": "这个新版接口的响应速度确实是非常之快。"},
    {"uttid": "33", "syn_text": "我们目前正在全力优化声码器的推理总耗时。"},
    {"uttid": "34", "syn_text": "这种人参果的口感很温润，非常适合你吃。"},
    {"uttid": "35", "syn_text": "实时流式音频生成是智能交互系统的核心关键。"},
    {"uttid": "36", "syn_text": "通过引入多奖励强化学习框架显著提升了表现。"},
    {"uttid": "37", "syn_text": "静态图分桶方案能够有效减少算子的各种开销。"},
    {"uttid": "38", "syn_text": "每一段旋律都仿佛在诉说着一个动人的故事。"},
    {"uttid": "39", "syn_text": "科学研究需要严谨的态度和坚持不懈的努力。"},
    {"uttid": "40", "syn_text": "城市夜晚的霓虹灯闪烁着五彩斑斓夺目的光芒。"},
    {"uttid": "41", "syn_text": "我们可以通过这个平台获取最新的全球行业资讯。"},
    {"uttid": "42", "syn_text": "今天的会议主要讨论公司未来一年的发展规划。"},
    {"uttid": "43", "syn_text": "阅读不仅能增长知识还能开阔我们的人生视野。"},
    {"uttid": "44", "syn_text": "这种全新的架构设计大大提升了系统的运行效率。"},

    # --- [第三组] 20-25字区间 (共15条，观察算力满载性能) ---
    {"uttid": "45", "syn_text": "他单独负责这项任务，从策划到执行亲力亲为，圆满完成。"},
    {"uttid": "46", "syn_text": "孩子们在操场上跑完步，纷纷跑到树荫下喝着凉开水休息。"},
    {"uttid": "47", "syn_text": "她专注地看着窗外，直到手机铃声突然响起才猛地回过神来。"},
    {"uttid": "48", "syn_text": "我看了疯狂元素城，里边关于水火元素谈恋爱的设定非常绝妙。"},
    {"uttid": "49", "syn_text": "每次熬煮小米粥时，奶奶总习惯加入一小把西洋参片调理身体。"},
    {"uttid": "50", "syn_text": "老街上的房屋参差错落，黛瓦与红色的砖墙相映成趣非常好看。"},
    {"uttid": "51", "syn_text": "她提前一周准备演讲稿，只为在参加校园辩论赛时能清晰表达。"},
    {"uttid": "52", "syn_text": "静态图分桶方案能够有效减少算子的各种任务调度和内存开销。"},
    {"uttid": "53", "syn_text": "本系统采用两阶段架构，首先使用大型模型生成语音符号序列。"},
    {"uttid": "54", "syn_text": "这种多奖励强化学习框架能够显著提升传统合成系统的表现力。"},
    {"uttid": "55", "syn_text": "我们致力于开发基于大型语言模型的高质量实时语音合成系统。"},
    {"uttid": "56", "syn_text": "实现更自然的情感表达和韵律控制是目前我们核心优化的目标。"},
    {"uttid": "57", "syn_text": "支持零样本语音克隆和极低延迟流式推理是这款模型的主要特性。"},
    {"uttid": "58", "syn_text": "这项技术报告已经在国际学术平台上发布并引起了专家的关注。"},
    {"uttid": "59", "syn_text": "灵活的推理方式能够支持多种采样策略以满足不同的业务场景。"}
]

# 创建保存目录
os.makedirs(f"{OUTPUT_DIR}/wavs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/headers", exist_ok=True)

print(f"开始测试，后端地址: {API_URL}\n" + "="*50)

summary_results = []

for item in test_items:
    uttid = item["uttid"]
    text = item["syn_text"]
    
    payload = {
        "voice_id": VOICE_ID,
        "input_text": text
    }
    
    print(f"正在处理 [{uttid}]: {text[:15]}...")
    
    start_time = time.perf_counter()
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        elapsed_script = time.perf_counter() - start_time
        
        if response.status_code == 200:
            wav_path = f"{OUTPUT_DIR}/wavs/{uttid}.wav"
            with open(wav_path, "wb") as f:
                f.write(response.content)
            
            header_path = f"{OUTPUT_DIR}/headers/{uttid}_info.txt"
            with open(header_path, "w") as f:
                for k, v in response.headers.items():
                    f.write(f"{k}: {v}\n")
            
            rtf = response.headers.get("x-rtf", "N/A")
            inf_time = response.headers.get("x-elapsed-seconds", "N/A")
            audio_len = response.headers.get("x-audio-seconds", "N/A")
            
            summary_results.append({
                "ID": uttid,
                "RTF": rtf,
                "推理时长": inf_time,
                "音频时长": audio_len,
                "脚本测得总耗时": f"{elapsed_script:.3f}s"
            })
            print(f" 成功! RTF: {rtf}")
        else:
            print(f" 失败! 状态码: {response.status_code}")
            
    except Exception as e:
        print(f" 请求异常: {e}")

# --- 打印最终汇总表格 ---
print("\n" + "="*80)
print(f"{'ID':<4} | {'RTF':<8} | {'推理时长':<10} | {'音频时长':<10} | {'脚本总耗时':<10}")
print("-" * 80)
for res in summary_results:
    print(f"{res['ID']:<4} | {res['RTF']:<8} | {res['推理时长']:<10} | {res['音频时长']:<10} | {res['脚本测得总耗时']:<10}")
