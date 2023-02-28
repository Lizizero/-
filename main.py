# 导入必要的库
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载GPT-2 模型
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 准备用户输入
user_input = input("你好，我是GPT-2，请问你有什么问题？")

# 将输入转换为token
token_input = tokenizer.encode(user_input, add_special_tokens=True)

# 将token转换为Tensor
token_input = torch.tensor([token_input])

# 模型预测
with torch.no_grad():
    output = model(token_input)
    logits = output[0]
    logits = logits[0, -1, :]
    logits /= 0.7

# 将预测结果转换为token
probs = torch.softmax(logits, dim=-1)
next_token_id = torch.multinomial(probs, num_samples=1).item()
generated_token = tokenizer.convert_ids_to_tokens([next_token_id])

# 打印聊天结果
print("GPT-2：", generated_token)
