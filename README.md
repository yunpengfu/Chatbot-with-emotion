# Chatbot-with-emotion

- 项目借鉴自[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)，当前项目只采用原作者的dialogue_model。
- 使用作者[GaoQ1](https://github.com/GaoQ1)提供的比较高质量的[闲聊数据集](https://pan.baidu.com/s/1v_JHQRUoT1VlKYFpWyOQ6Q)训练模型。（
- 本项目使用HuggingFace的transformers中BertTokenizer对语料Token，GPT2模型GPT2LMHeadModel对中文闲聊语料进行训练。
- classify.py拟用来训练对话文本情感分类，拟用来对用户的输入进行情感捕捉，加入到文本生成任务中，让闲聊机器人的输出具有与用户相匹配的情感。
