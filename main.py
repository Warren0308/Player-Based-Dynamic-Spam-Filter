import time
from threshold import g_config
from datetime import datetime
import json
import os
import numpy as np
import concurrent.futures
from collections import defaultdict, deque

if 1 == 1:
    os.system("pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org torch")
    os.system("pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org FlagEmbedding")
    os.system("pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org peft")
    os.system(
        "pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org faiss-cpu")
    os.system("pip install --upgrade accelerate -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com")
    os.system(
        "pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com pypi.org schedule")

from FlagEmbedding import BGEM3FlagModel
import faiss
import schedule
from threading import Thread

class InferenceService(object):
    def __init__(self, model_path=None):
        #self.model = torch.load('./bge_m3_complete_model.pth', map_location=torch.device('cpu'))
        self.model = BGEM3FlagModel('./bge_m3',  use_fp16=True)
        # 使用字典存储每个open_id在时间窗口内的聊天信息
        self.textDict = defaultdict(dict)
        self.threshold_lookup = ["first"] * 7 + ["second"] * 2 + ["third"] * 2 + ["fourth"] * 4 + ["fifth"] * 4 + \
                                ["sixth"] * 4 + ["seventh"] * 4 + ["eighth"] * 4 + ["ninth"] * 4 + ["tenth"]
    """获取对应scene id的资料
    {
        open_id: {
            scene_id: {
                flag: {
                    "index": faiss_index,
                    0: {
                        time1: {"start_time": None, "timestamps": deque([])},
                        time2: {"start_time": None, "timestamps": deque([])},
                        time3: {"start_time": None, "timestamps": deque([])},
                    },
                    message:{
                    }
                },
                flag:{
                }
            },
            scene_id:{
            }
        }
    }
    """

    def clean_dict_at_midnight(self):
        print("Cleaning dictionary...")
        self.textDict.clear()  # Clear the dictionary
        print("Dictionary cleaned!")

    def schedule_clean_up(self):
        # Schedule the clean-up task for 12 AM
        schedule.every().day.at("00:00").do(self.clean_dict_at_midnight)

        # Run the scheduler in a separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)  # Check every second

        Thread(target=run_scheduler, daemon=True).start()

    """消除旧文本，加入新文本"""
    def clean_expired_data(self, personal_data, similarity_faiss_idx, message_time, duration):
        flag_data = personal_data[similarity_faiss_idx][duration]["timestamps"]
        expired_time = message_time - duration
        # Clean deque and dict efficiently
        while flag_data and flag_data[0] < expired_time:
            flag_data.popleft()  # Pop from deque
        if flag_data:
            personal_data[similarity_faiss_idx][duration]["start_time"] = flag_data[0]
        else:
            personal_data[similarity_faiss_idx][duration]["start_time"] = 0
        return True

    """根据文本长度分类文本"""
    def len_calcul(self, text):
        l_content = len(text)
        return self.threshold_lookup[min(l_content, len(self.threshold_lookup) - 1)]

    """处理每个open_id的数据"""
    def vectorChecking(self, index_dict, vector):
        distances, indices = index_dict.search(vector, 1)
        nearest_idx = indices[0][0]
        highest_similarity = 1 - distances[0][0]  # Convert L2 distance to similarity
        return highest_similarity < 0.8, nearest_idx

    """文本向量化"""
    def message_vectorization(self, message):
        return self.model.encode(message, batch_size=15, max_length=50)['dense_vecs']

    """时间置换概念"""
    def time_management(self, personal_data, similarity_faiss_idx, message_time):
        # Define a helper function to process each time window
        def process_time_window(duration, value):
            start_time = value["start_time"]
            if message_time - start_time >= duration:
                self.clean_expired_data(personal_data, similarity_faiss_idx, message_time, duration)

        time_window_items = personal_data[similarity_faiss_idx].items()

        # Use ThreadPoolExecutor to parallelize
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks explicitly for more control
            futures = [
                executor.submit(process_time_window, duration, value)
                for duration, value in time_window_items
            ]
            concurrent.futures.wait(futures)
        return True

    def add_to_index(self, vector, personal_data):
        # Check if vector is 1D, then expand it to 2D
        if len(vector.shape) == 1:
            vector = np.expand_dims(vector, axis=0)
        key = vector.shape[1]
        # Initialize or retrieve the index efficiently
        index = personal_data.get("index", faiss.IndexFlatL2(key))
        faiss_idx = index.ntotal
        return index, faiss_idx, vector


    def process_request(self, request):
        """
        :param request:open_id&*#message&*#datetime(for test)
        :return: dict(message)
        """
        error = {0: "success"}
        result = {
            "open_id": "",  # 账号
            "scene_id": 0,  # 场景
            "spammingTime":{
                # duration:{ # 时段
                #     "no_similarity": 0,  # 相似文本的次数
                #     "threshold": 0,  # 阈值
                # },
            },
            "content": "",  # 文本内容
            "error": ""  # 出现问题才会有
        }

        try:
            request = request.decode("utf-8", "ignore")
        except:
            request = request

        # 空数据或数据不足不处理
        if request == "" or len(request.split('&*#')) < 3:
            result["error"] = "请求值有问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 数据分析
        # 账号 ｜ 内容
        data = request.strip().split("&*#")
        open_id, message, scene_id = data[0], data[1], data[2]
        result["open_id"] = open_id
        result["scene_id"] = scene_id
        result["content"] = message
        message_time = datetime.now().timestamp()
        # 测试包含时间(testing)
        if len(request.split("&*#")) > 3:
            message_time = request.strip().split("&*#")[-1]
            message_time = datetime.strptime(message_time, '%Y-%m-%d %H:%M:%S').timestamp()

        # 未配置场景不处理
        scene_data = g_config.get(int(scene_id), {})
        if not scene_data:
            result["error"] = "场景配置错误。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # 文本长度设定
        flag = self.len_calcul(message)

        # 文本向量化设定
        vector = self.message_vectorization(message)

        # 第一个数据所以不需要进行相似度分析（账号，场景和文本长度进行对比了解有没有这个数据）
        if not self.textDict.get(open_id, {}).get(scene_id, {}).get(flag, {}):
            self.textDict.setdefault(open_id, {}).setdefault(scene_id, {}).setdefault(flag, {})
            personal_data = self.textDict[open_id][scene_id][flag]
            index_dict, faiss_idx, vector = self.add_to_index(vector, personal_data)
            # Add the vector to the index
            index_dict.add(vector)
            personal_data["index"] = index_dict
            personal_data[faiss_idx] = {}
            # 把场景内设定的时段放入dict中(并根据设定时段输出结果）
            for duration, threshold in scene_data.items():
                personal_data[faiss_idx][duration] = {
                    "start_time": message_time,
                    "timestamps": deque([message_time])
                }
            # 获取第一个数据
            duration, threshold = next(iter(scene_data.items()))
            duration = str(int(duration / 60)) + "分钟"
            result["spammingTime"][duration] = {
                "no_similarity": 1,  # Similar text count
                "threshold": threshold  # Threshold
            }
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        personal_data = self.textDict[open_id][scene_id][flag]
        # 检查向量相似度
        index_dict, faiss_idx, vector = self.add_to_index(vector, personal_data)
        new, similarity_faiss_idx = self.vectorChecking(index_dict, vector)
        # 新文本就直接输出新的结果
        if new:
            index_dict.add(vector)
            personal_data["index"][vector.shape[1]] = index_dict
            personal_data[faiss_idx] = {}
            # 把场景内设定的时段放入dict中(并根据设定时段输出结果）
            for duration, threshold in scene_data.items():
                personal_data[faiss_idx][duration] = {
                    "start_time": message_time,
                    "timestamps": deque([message_time])
                }
            # 获取第一个数据
            duration, threshold = next(iter(scene_data.items()))
            duration = str(int(duration / 60)) + "分钟"
            result["spammingTime"][duration] = {
                "no_similarity": 1,  # Similar text count
                "threshold": threshold  # Threshold
            }
            result = json.dumps(result, ensure_ascii=False)
            return result, error
        else:
            # 清除旧数据
            time_data = self.time_management(personal_data, similarity_faiss_idx, message_time)
            if not time_data:
                result["error"] = "时间窗口出现问题。"
                result = json.dumps(result, ensure_ascii=False)
                return result, error
            # 添加新数据 & 数据返回成功
            for duration, threshold in scene_data.items():
                personal_data[similarity_faiss_idx][duration]["timestamps"].append(
                    message_time)
                no_similarity = len(
                    personal_data[similarity_faiss_idx][duration]["timestamps"])
                duration = str(int(duration / 60)) + "分钟"
                result["spammingTime"][duration] = {
                    "no_similarity": no_similarity,  # Similar text count
                    "threshold": threshold  # Threshold
                }
            duration = next(reversed(scene_data))
            result = json.dumps(result, ensure_ascii=False)
            return result, error


if __name__ == '__main__':
    # 使用示例
    model = InferenceService()
    model.schedule_clean_up()
    start_time = time.time()
    print("第1分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:00:00"))
    print("第2分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:01:00"))
    print("第3分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:02:00"))
    print("第4分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:03:00"))
    print("第5分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:04:00"))
    print("第6分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:05:00"))
    print("第7分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:06:00"))
    print("第8分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:07:00"))
    print("第9分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:08:00"))
    print("第10分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:09:00"))
    print("第11分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:10:00"))
    print("第12分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:11:00"))
    print("第13分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:12:00"))
    print("第14分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:13:00"))
    print("第15分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:14:00"))
    print("第16分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:15:00"))
    print("第17分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:16:00"))
    print("第18分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:17:00"))
    print("第19分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:18:00"))
    print("第20分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:19:00"))
    print("第21分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:20:00"))
    print("第22分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:21:00"))
    print("第23分钟: ", model.process_request(
        "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:22:00"))
    print("第24分钟: ", model.process_request(
        "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:23:00"))
    print("第25分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 17:24:00"))
    print("第30分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂群加加加32482933&*#2&*#2024-01-22 17:30:00"))
    print("第31分钟: ", model.process_request("28C073899E443002B6546B62BCC802&*#我叫黄胜衍&*#2&*#2024-01-22 17:31:00"))
    print("第32分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂群加加加32482933&*#2&*#2024-01-22 17:32:00"))
    print("第33分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂群加加加32482933&*#2&*#2024-01-22 17:33:00"))
    print("第36分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂qun加加加32482935&*#2&*#2024-01-22 17:36:00"))
    print("第66分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂qun加加加32482935&*#2&*#2024-01-22 18:06:00"))
    print("第67分钟: ",
          model.process_request("28C073899E443002B6546B62BCC802&*#外挂qun加加加32482935&*#2&*#2024-01-22 18:07:00"))
    print("第80分钟: ", model.process_request("28C073899E443002B6546B62BCC802&*#我叫黄胜衍&*#2&*#2024-01-22 18:20:00"))
    print("第120分钟: ",
          model.process_request(
              "12448278570560646234&*#＄1O0 二 8OK  Ruby,,Global，ADD，@（Whats）: +852 6619 6975&*#10&*#2024-01-22 19:24:00"))
    # 记录结束时间
    end_time = time.time()

    # 计算执行时间
    execution_time = end_time - start_time

    print(f"代码执行时间：{execution_time} 秒")