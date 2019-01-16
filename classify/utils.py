# 数据预处理
import re
# 读取csv
# 输入输出格式
# Input： CALL_ID	语音转文字结果	业务大类	业务小类
# 1521532934-208878	[{"end_time":"3.77","speech":"您好有什么可以帮您？","start_time":"0.92","target":"坐席"},
#                    {"end_time":"8.31","speech":"我这个信用卡丢了我先充值1下。","start_time":"3.77","target":"客户"},
#                    {"end_time":"10.80","speech":"您信用卡丢了是吗？","start_time":"8.31","target":"坐席"}]	卡片管理	卡片挂失
# CALL_ID content label_1 label2
def get_content(transcript):
    # 数字替换 金额替换
    content = ''
    for one_line in transcript:
        content += one_line['speech']
    
