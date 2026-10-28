# R13：完整 tile 的 mask 特化

这是可独立调用的 MSD 位布局组件，也是动态 mask 令 stride-2 load 退化的最小复现。尚未接入 vLLM、正式量化 frontend 或模型默认路径；也没有修改 Triton 编译器。

## 逐字节的例子

一行输入 `[0x12, 0x34, 0x56, 0x78]` 输出 `[0x31, 0x75, 0xca, 0x0e]`：相邻两个输入字节的高 nibble 合成一个输出字节，低 nibble 合成另一个输出字节并异或 `0x88`。所有高位结果放在前半行，低位结果放在后半行。整数运算和输出布局保持逐位一致。

kernel 用两个 uint16 载体处理这四个输入字节。它们的地址为 `base + 2*offset` 与 `base + 2*offset + 1`。原始实现始终给两次 load、两次 store 添加 `offset < K/4`。即使 `K=4096, BLOCK=1024`、逻辑 grid 为 `(M,1)`，旧编译路径也未必知道这个 mask 恒真，因而可能错过已有的连续读取后拆分偶/奇元素的优化。

## 改动和安全条件

`pack_msd` 根据真实行宽启动 `ceil((K/4)/BLOCK)` 个 tile：

- `K/4` 整除 `BLOCK` 时，所有 tile 都完整，编译期直接生成无 mask 的 load/store。
- 不能整除时，`tile < floor((K/4)/BLOCK)` 的 tile 使用无 mask 路径，最后一个 tile 保留原来的 mask。
- 小于一个 tile 的行始终走带 mask 路径。不存在“只有一个 tile 就删除 mask”的判断。

完整 tile 满足 `0 <= offset < K/4`；两个 uint16 输入地址都严格小于该行的 `K/2`，两个输出地址也都在本行之内。这里以 wrapper 启动的**逻辑 tile 数**为依据，与物理核数无关。直接用错误的 grid 调用私有 `_pack_msd` 不在接口合同内。

输入格式要求连续的二维 uint8 行、K 为 4 的倍数；输入和输出不共享存储。两字节对齐是 uint16 view 的要求。BLOCK 的范围是这个四 warp 实现的资源预算，不代表算法只能处理这些行宽。

## 验证

```python
import torch
import torch_npu
from third_party.ascend.tutorials.msd_pack import pack_msd

x = torch.randint(0, 256, (8, 4096), dtype=torch.uint8, device="npu")
y = pack_msd(x)
# 图捕获时提前分配 out，避免捕获分配行为。
pack_msd(x, out=y)
```

在仓库根目录执行：

```bash
python -m pytest third_party/ascend/unittest/pytest_ut/test_msd_pack_full_tile.py -q
python -m third_party.ascend.tutorials.msd_pack.benchmark
```

测试使用独立的逐字节 oracle，覆盖完整单 tile、多 tile、短尾块、尾部写保护、同地址修改输入后的图重放，以及所有单个 uint16 载体编码。后者不等于穷举两个载体的全部组合。

benchmark 对原始全 mask 实现与特化实现使用相同输入、输出、tile 大小和 100 节点 NPU 图，以事件测量每节点平均时间。结果是同址组件的图测量，包含图重放的摊销；不能当作独立 device trace 或模型 TPOT。具体测量值放在 PR 正文，原始日志留在源码树之外。

## 编译器侧还有什么没做

这个修补显式写出完整 tile 条件，复用编译器已有 deinterleave lowering。它没有实现新的 deinterleave 指令，也没有证明当前主线所有版本仍有同一慢路径。

自动化可以考虑两条路线：传递可验证、参与缓存键且启动时受校验的逻辑 grid 事实；或为可证明的完整 mask 生成快速分支，并保留尾块。前者不能把物理核数当逻辑 grid，后者需要评估分支、UB 和代码体积。这两条编译器方案均不包含在当前实现中。正式模型生产命中和端到端收益仍待单独证明。

当前安装版生成的完整 tile IR 中，两个偶/奇 load 各自连续读取一段再抽取元素，仍可能重复搬运；因此这也不是最小 HBM 字节数的实现。这里解决的是请求形状及其后续 lowering 退化，而非证明 pack 已达到访存下限。
