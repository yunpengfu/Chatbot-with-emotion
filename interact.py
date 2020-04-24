
import argparse
import logging
import transformers
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
import torch.nn.functional as F
import os
from datetime import datetime
import torch

def set_interact_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='使用GPU进行训练')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialogue_model_path', default='dialogue_model/model_epoch10/', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--emotion_length', default=50, type=int, required=False, help='获取情感倾向是截取长度')
    

    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def get_emotion_label(input_ids, args):
    pass




def main():
    args = set_interact_args()
    logger = create_logger(args)
    args.cuda = torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    # 加载tokenizer 
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    # 加载微调过的GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
    model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
        # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')
    while True:
        try:
            text = input("user:")
            samples_file.write("user:{}\n".format(text))
            history.append(tokenizer.encode(text,add_special_tokens=False))
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头 

            for history_id, history_utr in enumerate(history[-args.max_history_len:]): # 计数倒数第args.max_history_len到最后一个
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id) # 一句话的结尾才会加上[SEP]
                print(input_ids)    # 这个for是吧用户和机器人的每句话添加进input_ids
            # 在此对用户的输入进行情感取向判别
            """
            emotion = get_emotion_label(input_ids, args)
            """
            curr_input_tensor = torch.tensor(input_ids).long().to(device)
            generated = [ ]
            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model(input_ids=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            # 上面这个for是逐字生成，汇聚到generated
            # 我需要generated实际上包含多个句子，判断情感偏向，输出适当的句子
            # 另一个想法，采用beensearch再最后候选的n个句子中random一个输出
            """
            code code code
            """
            history.append(generated)
            text = tokenizer.convert_ids_to_tokens(generated)
            print("chatbot:" + "".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break

if __name__ == "__main__":
    PAD = '[PAD]'
    pad_id = 0
    main()