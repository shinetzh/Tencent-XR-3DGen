import pulsar
import time
import os
import json
import logging
from easydict import EasyDict as edict

class PulsarClient:
    
    def load_json(self,in_file):
        with open(in_file, encoding="utf-8") as f:
            data = json.load(f)
            return data
        
    def __init__(self, topic):
        try:
            cfg_json_name='./texture_generation/configs/client_texgen.json'
            if os.path.exists(cfg_json_name):
                json_data=self.load_json(cfg_json_name)
                configdict=edict(json_data)
                self.PULSAR_SERVICE_URL = configdict['uv_mcwy']['tdmq']['service_url']
                self.PULSAR_AUTH_TOKEN = configdict['uv_mcwy']['tdmq']['token']
                self.client = pulsar.Client(
                    authentication=pulsar.AuthenticationToken(self.PULSAR_AUTH_TOKEN),
                    service_url=self.PULSAR_SERVICE_URL
                )
            else:
                print("配置文件不存在")    
           
            # self.producer = self.client.create_producer(topic)
            # self.consumer = self.client.subscribe(topic, subscription_name='sub_topic1', message_listener=self.listener)
        except Exception as ex:
            print(f"获取pulsar环境变量参数出错了:{ex}")
     
     
    def send_async(self, message, callback, properties, partition_key):
        self.producer.send_async(message.encode('utf-8'), callback=callback, properties=properties, partition_key=partition_key)

    def listener(self, consumer, msg):
        print("Received message '{}' id='{}'".format(msg.data(), msg.message_id()))
        consumer.acknowledge(msg)

    def close(self):
        self.client.close()

if __name__ == "__main__":
    print("测试")
    pulsarClient=PulsarClient("test1")