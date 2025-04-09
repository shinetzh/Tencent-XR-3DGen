import rpyc
import time
import os

rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
rpyc_config["sync_request_timeout"] = None
conn = rpyc.connect('***.***.***.***', 8080) # 连接服务


image_path_list = []
image_dir = "/aigc_cfs_2/neoshang/data/test_humam_real"
for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)
    image_path_list.append(filepath)

for image_path in image_path_list:
    print(image_path)
    results = None
    while True:
        # available = conn.root.available()
        # if available:
        results, short_results = conn.root.image_to_caption(image_path)
        time.sleep(1)

        if results is not None:
            print(results)
            print(short_results)



# import rpyc
# rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
# rpyc_config["sync_request_timeout"] = None
# conn = rpyc.connect('***.***.***.***', 8080) # 连接服务

# image_path = ['/aigc_cfs/weixuan/code/LLaVA/llava/eval/images.jpeg', '/aigc_cfs/weixuan/code/captioner/cam-0050.png']
# # image_path : string or list of strings
# # max batch size: 10
# while True:
#     available = conn.root.available()
#     if available:
#         res = conn.root.image_to_caption(image_path)
#         try:
#             results, short_results = res
#         except:
#             print('failed')
#             break
#         break

# print(results) 
# print(short_results)
