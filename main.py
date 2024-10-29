import time
from threshold import g_config
from datetime import datetime
import json
import os
import numpy as np
from collections import defaultdict, deque
import concurrent.futures
from FlagEmbedding import BGEM3FlagModel
import faiss

class InferenceService(object):
    def __init__(self, model_path=None):
        #self.model = torch.load('./bge_m3_complete_model.pth', map_location=torch.device('cpu'))
        self.model = BGEM3FlagModel('./bge_m3',  use_fp16=True)
        # Use a dictionary to store each open_id's chat messages within the time window
        self.textDict = defaultdict(dict)
    """
    dictionary structure
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

    """delete the old message"""
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

    """assign to different message length"""
    def len_calcul(self, text):
        l_content = len(text)
        length_thresholds = [(6, "first"), (8, "second"), (10, "third"), (12, "fourth"),
                             (16, "fifth"), (20, "sixth"), (24, "seventh"), (28, "eighth"),
                             (32, "ninth"), (float('inf'), "tenth")]
        for threshold, flag in length_thresholds:
            if l_content <= threshold:
                return flag
        return False

    """find the most similarity message from the dictionary"""
    def vectorChecking(self, index_dict, vector):
        distances, indices = index_dict.search(vector, 1)
        nearest_idx = indices[0][0]
        highest_similarity = 1 - distances[0][0]  # Convert L2 distance to similarity
        return highest_similarity < 0.8, nearest_idx

    """message embedding by using bge-m3 model"""
    def message_vectorization(self, message):
        return self.model.encode(message, batch_size=15, max_length=50)['dense_vecs']

    """identify the old message duration"""
    def time_management(self, personal_data, similarity_faiss_idx, message_time):
        # Use ThreadPoolExecutor to parallelize time window checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for duration, value in personal_data[similarity_faiss_idx].items():
                start_time = value["start_time"]
                # If the message is still within the time window, skip cleaning
                if message_time - start_time >= duration:
                    # Submit the task to the thread pool
                    futures.append(executor.submit(self.clean_expired_data, personal_data, similarity_faiss_idx, message_time,duration))
            # Wait for all the threads to finish
            concurrent.futures.wait(futures)
        return True

    """add the new message embedding into the faiss"""
    def add_to_index(sele, vector, personal_data):
        # Check if vector is 1D, then expand it to 2D
        if len(vector.shape) == 1:
            vector = np.expand_dims(vector, axis=0)
        key = vector.shape[1]
        # Initialize or retrieve the index efficiently
        index = personal_data["index"].get(key, faiss.IndexHNSWFlat(key, 16))
        faiss_idx = index.ntotal
        return index, faiss_idx, vector


    def process_request(self, request):
        """
        :param request:open_id&*#message&*#datetime(for test)
        :return: dict(message)
        """
        error = {0: "success"}
        result = {
            "open_id": "",
            "scene_id": 0,
            "spammingTime":{
                # duration:{
                #     "no_similarity": 0,
                #     "threshold": 0,
                # },
            },
            "content": "",
            "error": ""
        }

        try:
            request = request.decode("utf-8", "ignore")
        except:
            request = request

        # No processing for empty or insufficient data
        if request == "" or len(request.split('&*#')) < 3:
            result["error"] = "请求值有问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error


        #open_id ｜ message ｜ scene_id
        data = request.strip().split("&*#")
        open_id, message, scene_id = data[0], data[1], data[2]
        result["open_id"] = open_id
        result["scene_id"] = scene_id
        result["content"] = message
        message_time = datetime.now().timestamp()
        # Existing the index 3 for testing. index 3 is the time
        if len(request.split("&*#")) > 3:
            message_time = request.strip().split("&*#")[-1]
            message_time = datetime.strptime(message_time, '%Y-%m-%d %H:%M:%S').timestamp()

        # No processing for empty message
        if message.strip() == "":
            result["error"] = "文本是空值。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # No processing for empty scene_id or not configured scene_id
        scene_data = g_config.get(int(scene_id), {})
        if not scene_data:
            result["error"] = "场景配置错误。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # Checking the message length
        flag = self.len_calcul(message)
        if not flag:
            result["error"] = "文本长度检测出现问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # Checking the message embeddings
        vector = self.message_vectorization(message)
        if vector is False:
            result["error"] = "文本向量化出现问题。"
            result = json.dumps(result, ensure_ascii=False)
            return result, error

        # No similarity analysis is required for the first data point (compare open_id, scene_id, and message to determine if this data exists).
        if not self.textDict.get(open_id, {}).get(scene_id, {}).get(flag, {}):
            self.textDict.setdefault(open_id, {}).setdefault(scene_id, {}).setdefault(flag, {"index": {}})
            personal_data = self.textDict[open_id][scene_id][flag]
            index_dict, faiss_idx, vector = self.add_to_index(vector, personal_data)
            # Add the vector to the index
            index_dict.add(vector)
            personal_data["index"][vector.shape[1]] = index_dict
            personal_data[faiss_idx] = {}
            # Place the specified time intervals within the scene into a dictionary (and output results according to the specified intervals).
            for duration, threshold in scene_data.items():
                personal_data[faiss_idx][duration] = {
                    "start_time": message_time,
                    "timestamps": deque([message_time])
                }
            # Store each duration time
            duration, threshold = next(iter(scene_data.items()))
            duration = str(int(duration / 60)) + "分钟"
            result["spammingTime"][duration] = {
                "no_similarity": 1,  # Similar text count
                "threshold": threshold  # Threshold
            }
            result = json.dumps(result, ensure_ascii=False)
            return result, error
        personal_data = self.textDict[open_id][scene_id][flag]
        # Checking the similarity embeddings
        index_dict, faiss_idx, vector = self.add_to_index(vector, personal_data)
        new, similarity_faiss_idx = self.vectorChecking(index_dict, vector)
        # "New embedding" refers to a representation that does not match any existing messages in the dictionary.
        if new:
            index_dict.add(vector)
            personal_data["index"][vector.shape[1]] = index_dict
            personal_data[faiss_idx] = {}
            # Place the specified time intervals within the scene into a dictionary (and output results according to the specified intervals).
            for duration, threshold in scene_data.items():
                personal_data[faiss_idx][duration] = {
                    "start_time": message_time,
                    "timestamps": deque([message_time])
                }
            # Store each duration time
            duration, threshold = next(iter(scene_data.items()))
            duration = str(int(duration / 60)) + "分钟"
            result["spammingTime"][duration] = {
                "no_similarity": 1,  # Similar text count
                "threshold": threshold  # Threshold
            }
            result = json.dumps(result, ensure_ascii=False)
            return result, error
        else:
            # Checking the time stored in the dictionary
            time_data = self.time_management(personal_data, similarity_faiss_idx, message_time)
            if not time_data:
                result["error"] = "时间窗口出现问题。"
                result = json.dumps(result, ensure_ascii=False)
                return result, error
            # Store each duration time
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
    # Instances
    model = InferenceService()
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