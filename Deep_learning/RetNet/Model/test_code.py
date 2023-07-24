import torch
from retnet import RetNetModel, RetNetModelWithLMHead, RetNetConfig

torch.manual_seed(0)
config = RetNetConfig(num_layers=8, vocab_size=100, hidden_size=512, num_heads=4, use_default_gamma=False, chunk_size=4)
model = RetNetModel(config)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(model.blocks[0].msr.decay)

input_ids = torch.LongTensor([[1,2,3,4, 1,2,3,4]]).to(device)

out, parallel_past_kv = model(input_ids, forward_impl='parallel', return_kv=True)

past_kv = None
rnn_outs = []

for i in range(input_ids.shape[1]):
    rnn_out, past_kv = model(input_ids[:, i:i+1], forward_impl='recurrent', past_kv=past_kv, return_kv=True, sequence_offset=i)
    rnn_outs.append(rnn_out)

rnn_outs = torch.cat(rnn_outs, dim=1)

chunk_out, chunk_past_kv = model(input_ids, forward_impl='chunkwise', return_kv=True)

"""
torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool
此函数检查所有 input 和 other 输入是否满足条件
|input - other| ≤ atol + rtol × |other|
"""
print(torch.allclose(out, rnn_outs, atol=1e-5))
print(torch.allclose(out, chunk_out, atol=1e-5))
print(torch.allclose(rnn_outs, chunk_out, atol=1e-5))

for i, (p, r, c) in enumerate(zip(parallel_past_kv, past_kv, chunk_past_kv)):
    print(f"layer: {i + 1}")
    print(torch.allclose(p, r, atol=1e-5))
    print(torch.allclose(p, c, atol=1e-5))
    print(torch.allclose(r, c, atol=1e-5))


torch.manual_seed(0)
config = RetNetConfig(num_layers=8, vocab_size=100, hidden_size=512, num_heads=4, use_default_gamma=False, chunk_size=4)
model = RetNetModelWithLMHead(config).to(device)
model.eval()

p_generated = model.generate(input_ids, parallel_compute_prompt=True, max_new_tokens=20, do_sample=False, early_stopping=False)
r_generated = model.generate(input_ids, parallel_compute_prompt=False, max_new_tokens=20, do_sample=False, early_stopping=False)

print('p_generated is: ', p_generated)
print('r_generated is: ', r_generated)