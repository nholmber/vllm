import torch
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch
from vllm._aiter_ops import rocm_aiter_ops
torch.manual_seed(0)
dev='cuda:0'; hc,H=4,4096; hc_mult3=hc*2+hc*hc
args=(1e-6,1e-6,1e-6,2.0,20)
def rel(ref,b):
    ref=ref.float(); b=b.float()
    return (ref-b).abs().max().item()/(ref.abs().max().item()+1e-9), torch.isfinite(b).all().item()
print(f"{'T':>5} {'aiter_rel':>10} {'aiter_fin':>9}")
for T in (1,4,16,64,256,1024):
    residual=torch.randn(T,hc,H,device=dev,dtype=torch.bfloat16)
    fn=torch.randn(hc_mult3,hc*H,device=dev,dtype=torch.float32)*0.02
    sc=torch.ones(3,device=dev,dtype=torch.float32); ba=torch.zeros(hc_mult3,device=dev,dtype=torch.float32)
    pm_t,cm_t,li_t=mhc_pre_torch(residual,fn,sc,ba,*args)
    ar,af=rel(li_t, rocm_aiter_ops.mhc_pre(residual,fn,sc,ba,*args)[2])
    print(f"{T:>5} {ar:>10.3e} {str(af):>9}")
