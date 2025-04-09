cp huggingface.tar.gz ~/.cache/ && cd ~/.cache/ && tar -zxvf huggingface.tar.gz && rm huggingface.tar.gz
cd /data/code 
python init_redis.py -- 0
python interface_redis_normal.py

