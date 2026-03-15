This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where line numbers have been added, security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
cuda/
  clifford_thrust-0.0.1-cp310-cp310-linux_x86_64.whl
  clifford_thrust-0.0.1-cp312-cp312-linux_x86_64.whl
  whl.md
ARCHITECTURE_DETAILED.md
ARCHITECTURE.md
gffn.py
model.py
networks.py
README.md
train.py
utils.py
```

# Files

## File: cuda/whl.md
````markdown
1: 
````

## File: ARCHITECTURE_DETAILED.md
````markdown
 1: ```mermaid
 2:     graph TD;
 3:     A[Client] -->|User Input| B{Application};
 4:     B --> C[API Gateway];
 5:     C -->|Request| D[Microservice 1];
 6:     C -->|Request| E[Microservice 2];
 7:     D --> F[Database];
 8:     E --> F;
 9:     F -->|Data| D;
10:     F -->|Data| E;
11:     B --> G[Cache];
12:     G -->|Cache Lookup| D;
13:     G -->|Cache Lookup| E;
14: ``` 
15: 
16: This diagram represents the CliffordNet system design, illustrating the flow of data between the various components, including the client, application, API gateway, microservices, and the database with cache integration.
````

## File: ARCHITECTURE.md
````markdown
1: ```mermaid
2: graph TD;
3:     A[Patch Embedding] --> B[CliffordNet Blocks];
4:     B --> C[Clifford Interaction Layers];
5:     C --> D[Output Classification Head];
6: ```
````

## File: gffn.py
````python
 1: import torch
 2: import torch.nn as nn
 3: import torch.nn.functional as F
 4: from utils import DropPath
 5: from clifford_thrust import LayerNorm2d, CliffordInteraction        
 6: 
 7: 
 8: class gFFN(nn.Module):
 9:     """
10:     Geometric Feed-Forward Network (gFFN)
11:     
12:     A standalone module that replaces standard MLP with Clifford Geometric Interactions.
13:     Can be used as a plug-and-play component in various backbones.
14:     
15:     Args:
16:         dim (int): Feature dimension.
17:         cli_mode (str): 'full', 'wedge', 'inner'.
18:         ctx_mode (str): 'diff' (Laplacian) or 'abs'.
19:         gffn_mode (str): 
20:             - 'l': Local only (Conv context).
21:             - 'g': Global only (GlobalAvg context).
22:             - 'h': Hybrid (Local + Beta * Global).
23:         shifts (list): Shifts for rolling interaction.
24:     """
25:     def __init__(self, dim, cli_mode='full', ctx_mode='diff', gffn_mode='h', 
26:                  shifts=[1, 2, 4], drop_path=0., init_values=1e-5, enable_cuda=False):
27:         super().__init__()
28:         
29:         self.dim = dim
30:         self.cli_mode = cli_mode  
31:         self.ctx_mode = ctx_mode 
32:         self.gffn_mode = gffn_mode.lower()
33:         
34:         self.norm = LayerNorm2d(dim)
35:         InteractionLayer = CliffordInteraction 
36:             
37:         self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
38: 
39:         if self.gffn_mode in ['l', 'h']:
40:             self.get_context_local = nn.Sequential(
41:                 nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
42:                 nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
43:                 nn.BatchNorm2d(dim),
44:                 nn.SiLU()            
45:             )
46:             self.inter_local = InteractionLayer(dim, cli_mode=cli_mode, ctx_mode=ctx_mode, shifts=shifts)
47:         else:
48:             self.get_context_local = None
49:             self.inter_local = None
50: 
51:         if self.gffn_mode in ['g', 'h']:
52:             self.inter_global = InteractionLayer(dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2])
53:         else:
54:             self.inter_global = None
55: 
56:         if self.gffn_mode == 'h':
57:             self.beta = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
58: 
59:         self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
60:         self.gamma = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
61:         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()        
62:         
63:     def forward(self, x):
64:         shortcut = x
65:         x_ln = self.norm(x) 
66:         z_state = self.get_state(x_ln)
67:         g_feat_total = None
68:         
69:         if self.get_context_local is not None:
70:             z_ctx_local = self.get_context_local(x_ln)
71:             g_feat_local = self.inter_local(z_state, z_ctx_local)
72:         
73:         if self.inter_global is not None:
74:             z_ctx_global = x_ln.mean(dim=[-2, -1], keepdim=True).expand_as(x_ln)                
75:             g_feat_global = self.inter_global(z_state, z_ctx_global)
76: 
77:         if self.gffn_mode == 'l':
78:             g_feat_total = g_feat_local
79:         elif self.gffn_mode == 'g':
80:             g_feat_total = g_feat_global
81:         elif self.gffn_mode == 'h':
82:             g_feat_total = g_feat_local + self.beta * g_feat_global
83:             
84:         combined = torch.cat([x_ln, g_feat_total], dim=1)
85:         gate = torch.sigmoid(self.gate_fc(combined))
86:         
87:         x_mixed = F.silu(x_ln) + gate * g_feat_total
88:         x = shortcut + self.drop_path(self.gamma * x_mixed)
89:         
90:         return x
````

## File: model.py
````python
  1: import torch
  2: import torch.nn as nn
  3: import torch.nn.functional as F
  4: from utils import DropPath
  5: 
  6: try:
  7:     from clifford_thrust import LayerNorm2d, CliffordInteraction
  8:     print("✅ Successfully loaded accelerated Clifford kernels (CUDA).")
  9:     has_acceleration = True
 10: except ImportError:
 11:     print("⚠️ 'clifford_thrust' not found. Falling back to pure PyTorch implementation (Slower).")
 12:     has_acceleration = False
 13: 
 14: class LayerNorm2d_PyTorch(nn.Module):
 15:     def __init__(self, num_channels, eps=1e-6):
 16:         super().__init__()
 17:         self.weight = nn.Parameter(torch.ones(num_channels))
 18:         self.bias = nn.Parameter(torch.zeros(num_channels))
 19:         self.eps = eps
 20: 
 21:     def forward(self, x):
 22: 
 23:         u = x.mean(1, keepdim=True)
 24:         s = (x - u).pow(2).mean(1, keepdim=True)
 25:         x = (x - u) / torch.sqrt(s + self.eps)
 26:         x = self.weight[:, None, None] * x + self.bias[:, None, None]
 27:         return x
 28:     
 29: class CliffordInteraction_PyTorch(nn.Module):
 30:     """
 31:     Args:
 32:         cli_mode: 'full', 'wedge', 'inner'
 33:         ctx_mode: 
 34:             - 'diff': C = C_local - H (discrete Laplacian)
 35:             - 'abs' : C = C_local
 36:             - 'others': to be added
 37:     """    
 38:     def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2]):
 39:         super().__init__()
 40:         self.dim = dim
 41:         self.cli_mode = cli_mode  
 42:         self.ctx_mode = ctx_mode 
 43:         self.act = nn.SiLU()
 44:         
 45:         self.shifts = shifts
 46:         self.shifts = [s for s in self.shifts if s < dim]
 47:         
 48:         self.branch_dim = dim * len(self.shifts)
 49:         
 50:         self.proj_ = None
 51:         
 52:         if self.cli_mode == 'full':
 53:             cat_dim = self.branch_dim*2
 54:         elif self.cli_mode in ['wedge', 'inner']:
 55:             cat_dim = self.branch_dim
 56:         else:
 57:             raise ValueError(f"Invalid cli_mode: {cli_mode}")
 58:         self.proj_ = nn.Conv2d(cat_dim, dim, kernel_size=1)    
 59: 
 60:     def forward(self, z1, z2):
 61:         
 62:         if self.ctx_mode == 'diff':
 63:             C = z2 - z1  
 64:         elif self.ctx_mode == 'abs':
 65:             C = z2
 66:             
 67:         feats = []
 68:         for s in self.shifts:
 69:             C_shifted = torch.roll(C, shifts=s, dims=1)     
 70:             if self.cli_mode in ['wedge', 'full']:
 71:                 z1_shifted = torch.roll(z1, shifts=s, dims=1)
 72:                 wedge = z1 * C_shifted - C * z1_shifted
 73:                 feats.append(wedge)
 74:             if self.cli_mode in ['inner', 'full']:
 75:                 inner = self.act(z1 * C_shifted)
 76:                 feats.append(inner)
 77:         x_ = torch.cat(feats, dim=1)
 78:         out = self.proj_(x_)
 79:         return out  
 80: 
 81: class CliffordAlgebraBlock(nn.Module):
 82:     def __init__(self, dim, cli_mode='full', ctx_mode = 'diff', shifts=[1, 2], enable_gFFNG=False, num_heads=1, mlp_ratio=0., drop=0., drop_path=0.1, init_values=1e-5, enable_cuda=False):
 83:         super().__init__()
 84: 
 85:         self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
 86:         self.get_context_local = nn.Sequential(
 87:             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
 88:             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
 89:             nn.BatchNorm2d(dim),
 90:             nn.SiLU()            
 91:          )
 92:         self.enable_gFFNG = enable_gFFNG
 93:         if enable_cuda:
 94:             self.norm = LayerNorm2d(dim)
 95:             self.clifford_interaction_local = CliffordInteraction(dim, cli_mode, ctx_mode, shifts)
 96:             if self.enable_gFFNG:
 97:                 self.clifford_interaction_global = CliffordInteraction(dim, cli_mode='full', ctx_mode='diff', shifts=[1,2])
 98:         else:
 99:             self.norm = LayerNorm2d_PyTorch(dim)
100:             self.clifford_interaction_local = CliffordInteraction_PyTorch(dim, cli_mode, ctx_mode, shifts) 
101:             if self.enable_gFFNG:
102:                 self.clifford_interaction_global = CliffordInteraction_PyTorch(dim, cli_mode='full', ctx_mode='diff', shifts=[1,2])
103: 
104:         self.act = nn.SiLU()
105:         self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
106:         self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
107:         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
108:         
109:     def forward(self, x):
110:         shortcut = x
111:         x_ln = self.norm(x) 
112:         
113:         z_state = self.get_state(x_ln)
114:         z_context_local = self.get_context_local(x_ln)
115:         g_feat_total = self.clifford_interaction_local(z_state, z_context_local)
116:         
117:         if self.enable_gFFNG:
118:             z_context_global = x_ln.mean(dim=[-2, -1], keepdim=True).expand_as(x_ln)
119:             g_feat_global = self.clifford_interaction_global(z_state, z_context_global)
120:             g_feat_total = g_feat_total + g_feat_global
121:         
122:         combined = torch.cat([x_ln, g_feat_total], dim=1)
123:         gate = torch.sigmoid(self.gate_fc(combined))
124:         x_mixed = F.silu(x_ln) + gate * g_feat_total
125:         x = shortcut + self.drop_path(self.gamma * x_mixed)
126:         
127:         return x    
128: 
129: class GeometricStem(nn.Module):
130:     def __init__(self, in_chans=3, embed_dim=128, patch_size=2):
131:         super().__init__()
132:         
133:         if patch_size == 1:
134:             self.proj = nn.Sequential(
135:                 nn.Conv2d(in_chans, embed_dim // 2, 3, stride=1, padding=1, bias=False),
136:                 nn.BatchNorm2d(embed_dim // 2),
137:                 nn.SiLU(),
138:                 nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=1, padding=1, bias=False),
139:             )
140:         elif patch_size == 2:
141:             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
142:             
143:         elif patch_size == 4:
144:             self.proj = nn.Sequential(
145:                 nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
146:                 nn.BatchNorm2d(embed_dim // 2),
147:                 nn.SiLU(),
148:                 nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
149:             )
150:         else:
151:             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
152: 
153:         self.norm = nn.BatchNorm2d(embed_dim) 
154: 
155:     def forward(self, x):
156:         x = self.proj(x)
157:         x = self.norm(x)
158:         return x
159:     
160: class CliffordNet(nn.Module):
161:     def __init__(self, num_classes=10, patch_size=4, embed_dim=128, cli_mode='full', ctx_mode='diff', shifts=[1, 2], depth=6, num_heads=1, mlp_ratio=0., drop_rate=0., drop_path_rate=0.1,enable_cuda=False):
162:         super().__init__()
163:         
164:         self.patch_embed = GeometricStem(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)
165:         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
166:         
167:         self.blocks = nn.ModuleList([
168:             CliffordAlgebraBlock(
169:                 dim=embed_dim, 
170:                 cli_mode=cli_mode,
171:                 ctx_mode=ctx_mode,
172:                 shifts=shifts,                
173:                 num_heads=num_heads, 
174:                 drop_path=dpr[i],
175:                 enable_cuda=enable_cuda,
176:             )
177:             for i in range(depth)
178:         ])
179:         
180:         self.norm = nn.LayerNorm(embed_dim)
181:         self.head = nn.Linear(embed_dim, num_classes)
182:         self.apply(self._init_weights)
183: 
184:     def _init_weights(self, m):
185:         if isinstance(m, (nn.Conv2d, nn.Linear)):
186:             nn.init.trunc_normal_(m.weight, std=.02)
187:             if m.bias is not None:
188:                 nn.init.constant_(m.bias, 0)
189:         elif isinstance(m, nn.LayerNorm):
190:             nn.init.constant_(m.bias, 0)
191:             nn.init.constant_(m.weight, 1.0)
192: 
193:     def forward_features(self, x):
194: 
195:         x = self.patch_embed(x) 
196:         for block in self.blocks:
197:             x = block(x) 
198:         return x
199: 
200:     def forward(self, x):
201:         
202:         x = self.forward_features(x)
203:         x = x.mean(dim=[-2, -1]) 
204:         x = self.norm(x)
205:         x = self.head(x)
206:         return x
````

## File: networks.py
````python
  1: from model import CliffordNet
  2:     
  3:     
  4: def gen_shifts(n):
  5:     return [1 << i for i in range(n)]
  6: 
  7: def gen_shifts_fibonacci(n):
  8:     a, b = 1, 2
  9:     for _ in range(n):
 10:         yield a
 11:         a, b = b, a + b
 12:         
 13: def cliffordnet_12_2(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
 14:     # Nano: shifts=[1, 2]
 15:     shifts = gen_shifts(2)
 16:     return CliffordNet(
 17:         enable_cuda=enable_cuda,
 18:         num_classes=num_classes,
 19:         patch_size=patch_size,
 20:         embed_dim=embed_dim, 
 21:         cli_mode='full', 
 22:         ctx_mode='diff',
 23:         shifts=shifts, 
 24:         depth=12,
 25:         drop_path_rate=0.3
 26:     )    
 27: 
 28: def cliffordnet_12_3(num_classes=100, patch_size=1, embed_dim=160, enable_cuda=False):
 29:     # Nano: shifts=[1, 2]
 30:     shifts = gen_shifts(3)
 31:     return CliffordNet(
 32:         enable_cuda=enable_cuda,
 33:         num_classes=num_classes,
 34:         patch_size=patch_size,
 35:         embed_dim=embed_dim, 
 36:         cli_mode='full', 
 37:         ctx_mode='diff',
 38:         shifts=shifts, 
 39:         depth=12,
 40:         drop_path_rate=0.3 
 41:     )    
 42: 
 43: def cliffordnet_12_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
 44:     # Lite: shifts=[1, 2, 4, 8, 16]
 45:     shifts = gen_shifts(5)
 46:     return CliffordNet(
 47:         enable_cuda=enable_cuda,
 48:         num_classes=num_classes,
 49:         patch_size=patch_size,
 50:         embed_dim=embed_dim, 
 51:         cli_mode='full', 
 52:         ctx_mode='diff',
 53:         shifts=shifts, 
 54:         depth=12,
 55:         drop_path_rate=0.3 
 56:     )    
 57: 
 58: def cliffordnet_18_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
 59:     # Lite: shifts=[1, 2, 4, 8, 16]
 60:     shifts = gen_shifts(5)
 61:     return CliffordNet(
 62:         enable_cuda=enable_cuda,
 63:         num_classes=num_classes,
 64:         patch_size=patch_size,
 65:         embed_dim=embed_dim, 
 66:         cli_mode='full', 
 67:         ctx_mode='diff',
 68:         shifts=shifts, 
 69:         depth=18,
 70:         drop_path_rate=0.3 
 71:     )  
 72: 
 73: def cliffordnet_32_3(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
 74:     # Small: 32 layers
 75:     # Shifts: [1, 2, 4]
 76:     shifts = gen_shifts(3)
 77:     return CliffordNet(
 78:         enable_cuda=enable_cuda,
 79:         num_classes=num_classes,
 80:         patch_size=patch_size,
 81:         embed_dim=embed_dim, 
 82:         cli_mode='full', 
 83:         ctx_mode='diff',
 84:         shifts=shifts, 
 85:         depth=32,
 86:         drop_path_rate=0.3 
 87:     )     
 88: 
 89: def cliffordnet_32_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
 90:     # Small: 32 layers
 91:     # Shifts: [1, 2, 4, 8, 16]
 92:     shifts = gen_shifts(5)
 93:     return CliffordNet(
 94:         enable_cuda=enable_cuda,
 95:         num_classes=num_classes,
 96:         patch_size=patch_size,
 97:         embed_dim=embed_dim, 
 98:         cli_mode='inner', 
 99:         ctx_mode='diff',
100:         shifts=shifts, 
101:         depth=32,
102:         drop_path_rate=0.3 
103:     )     
104:  
105: def cliffordnet_64_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
106:     # Deep: 64 layers
107:     # Shifts: [1, 2, 4, 8, 16]
108:     shifts = gen_shifts(5)
109:     return CliffordNet(
110:         enable_cuda=enable_cuda,
111:         num_classes=num_classes,
112:         patch_size=patch_size,
113:         embed_dim=embed_dim, 
114:         cli_mode='inner', 
115:         ctx_mode='diff',
116:         shifts=shifts, 
117:         depth=64,
118:         drop_path_rate=0.4 
119:     )
````

## File: README.md
````markdown
  1: <div align="center">
  2: 
  3: # CliffordNet: All You Need is Geometric Algebra
  4:   
  5: 
  6: 
  7: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  8: [![Github](https://img.shields.io/badge/Github-grey?logo=github)](https://github.com)
  9: [![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
 10: [![arXiv](https://img.shields.io/badge/arXiv-2601.06793-b31b1b.svg)](https://arxiv.org/abs/2601.06793)
 11: [![Hardware](https://img.shields.io/badge/Triton-Accelerated-blue)](https://triton-lang.org/)
 12: 
 13: “The two systems [Hamilton’s and Grassmann’s]
 14: are not only consistent with one another, but they
 15: are actually parts of a larger whole.”
 16: 
 17: — William Kingdon Clifford, 1878
 18: 
 19: </div>
 20: 
 21: Official implementation of the paper **"CliffordNet: All You Need is Geometric Algebra"**.
 22: 
 23: We introduce **Clifford Algebra Network (CAN)**, a novel vision backbone that challenges the necessity of Feed-Forward Networks (FFNs) in deep learning. By operationalizing the full **Clifford Geometric Product** ($uv = u \cdot v + u \wedge v$), we unify feature coherence and structural variation into a single, algebraically complete interaction layer.
 24: 
 25: Our **"No-FFN"** variant demonstrates that this geometric interaction is so expressive that heavy MLPs become redundant, establishing a new Pareto frontier for efficient visual representation learning.
 26: 
 27: ## 🚀 News & Updates
 28: 
 29: *   **[2026-02-17]** 🔥 **Released the code for preliminary experiments on CIFAR-100.**
 30: *   **[2026-01-20]** 🏆 **New SOTA:**
 31:     *   **Nano (1.4M)** reaches **77.82%**, outperforming ResNet-18 (11M).
 32:     *   **Lite (2.6M)** reaches **79.05%** without FFN, rivaling ResNet-50.
 33:     *   **32-Layer Deep Model** achieves **81.42%** with only 4.8M parameters.
 34: *   **[2026-01-12]** ⚡ **Performance Preview:** We have successfully implemented a custom **Fused Triton Kernel** for the Clifford Interaction layer. Preliminary benchmarks on RTX 4090 show a **10x kernel speedup** and **~2x end-to-end training speedup**. *Code coming soon!*
 35: *   **[2026-01-01]** 🏆 **SOTA on CIFAR-100:** Our Nano model (1.4M) matches ResNet-18 (11M), and our No-FFN model outperforms MobileNetV2 by >6%.
 36: 
 37: ## 🏆 Main Results (CIFAR-100)
 38: 
 39: We compare CliffordNet against established efficient backbones under a rigorous "Modern Training Recipe" (200 Epochs, AdamW, AutoAugment, DropPath).
 40: 
 41: ### Efficiency & Performance
 42: | Model Variant | Params | MLP Ratio | Context Mode | Top-1 Acc | vs. Baseline |
 43: | :--- | :---: | :---: | :---: | :---: | :--- |
 44: | **Baselines** | | | | | |
 45: | MobileNetV2 | 2.3M | - | - | 70.90% | - |
 46: | ShuffleNetV2 1.5x | 2.6M | - | - | 75.95% | - |
 47: | ResNet-18 | 11.2M | - | - | 76.75% | - |
 48: | ResNet-50 | 23.7M | - | - | 79.14% | - |
 49: | **CliffordNet (Ours)** | | | | | |
 50: | **CAN-Nano** | **1.4M** | **0.0** | Diff ($\Delta H$) | **77.82%** | <span style="color:green">> ResNet-18</span> |
 51: | **CAN-Lite** | **2.6M** | **0.0** | Diff ($\Delta H$) | **79.05%** | <span style="color:green">~ ResNet-50</span> |
 52: | **CAN-32 (Deep)**| 4.8M | 0.0 | Full | **81.42%** | <span style="color:green">**SOTA**</span> |
 53: | **CAN-64 (Deep)**| 8.6M | 0.0 | Full | **82.46%** | <span style="color:green">**SOTA**</span> |
 54: 
 55: > **Key Insight:** Our **Nano** variant (1.4M) outperforms the heavy-weight **ResNet-18** (11.2M) by **+1.07%** while using **$8\times$ fewer parameters**. The **Lite** variant (No-FFN) effectively matches ResNet-50 with **$9\times$ fewer parameters**.
 56: 
 57: ## 🏗️ Architecture & Theory
 58: 
 59: The evolution of features in CliffordNet is governed by a **Geometric Diffusion-Reaction Equation**. We introduce a unified superposition principle that integrates local differential context and global mean fields:
 60: 
 61: $$
 62: \frac{\partial H}{\partial t} = \mathcal{P}_{loc}\Big( H (\mathcal{C}_{loc}) \Big) + \beta \cdot \mathcal{P}_{glo}\Big( H (\mathcal{C}_{glo}) \Big) 
 63: $$
 64: 
 65: Where $\mathcal{C}_{loc} \approx \Delta H$ (Local Laplacian) and $\mathcal{C}_{glo} = \text{GlobalAvg}(H)$. The interaction term is expanded via the **Clifford Geometric Product**, unifying scalar and bivector components:
 66: 
 67: $$
 68: \mathcal{P}\Big( H(\mathcal{C}) \Big) = \mathcal{P}\Big( \underbrace{\mathcal{D}(H, \mathcal{C})}_{\text{Scalar Component}} \oplus \underbrace{\mathcal{W}(H, \mathcal{C})}_{\text{Bivector Component}} \Big)
 69: $$
 70: 
 71: ## 🛠️ Usage
 72: 
 73: CliffordNet supports two execution modes: a **High-Performance Mode** (using custom CUDA kernels) and a **Compatibility Mode** (pure PyTorch).
 74: 
 75: Requirements:
 76: 
 77: ```
 78: torch>=2.0.0
 79: python>=3.10
 80: ```
 81: 
 82: ### 1. Installation (Hardware Acceleration)
 83: Install the compiled `clifford_thrust` wheel matching your environment。
 84: 
 85: > ⚠️ **Note:** The provided wheels are currently optimized and tested specifically for **NVIDIA RTX 4090** (Compute Capability 8.9). For other GPUs, please use the standard PyTorch mode.
 86: 
 87: *   **Python 3.10 + CUDA 11.8**
 88:     ```bash
 89:     pip install cuda/clifford_thrust-0.0.1-cp310-cp310-linux_x86_64.whl
 90:     ```
 91: 
 92: *   **Python 3.12 + CUDA 12.8**
 93:     ```bash
 94:     pip install cuda/clifford_thrust-0.0.1-cp312-cp312-linux_x86_64.whl
 95:     ```
 96: 
 97: ### 2. Training
 98: 
 99: To launch training, simply run the script. The code automatically handles the fallback if the accelerated kernels are not installed.
100: 
101: *   **Accelerated Mode (Recommended):**
102:     Requires `clifford_thrust` installed.
103:     ```bash
104:     python train.py --enable_cuda
105:     ```
106: 
107: *   **Standard Mode (Pure PyTorch):**
108:     Works on any device (MPS/CUDA) without extra dependencies.
109:     ```bash
110:     python train.py
111:     ```
112: 
113: ### 3. Python API & Model Zoo
114: 
115: You can instantiate the models directly using the `CliffordNet` class. Below are the configurations for our top-performing variants.
116: 
117: ```python
118: from model import CliffordNet
119: 
120: # ---------------------------------------------------------
121: # 1. CliffordNet-Nano (1.4M)
122: # ---------------------------------------------------------
123: model_nano = CliffordNet(
124:     num_classes=100,
125:     patch_size=2,
126:     embed_dim=128,
127:     depth=12,
128:     cli_mode='full',
129:     ctx_mode='diff',
130:     shifts=[1, 2],
131:     drop_path_rate=0.3
132: )
133: 
134: # ---------------------------------------------------------
135: # 2. CliffordNet-Lite (2.6M)
136: # ---------------------------------------------------------
137: model_lite = CliffordNet(
138:     num_classes=100,
139:     patch_size=2,
140:     embed_dim=128,
141:     depth=12,
142:     cli_mode='full',
143:     ctx_mode='diff',
144:     shifts=[1, 2, 4, 8, 16], 
145:     drop_path_rate=0.3
146: )
147: ```
148: 
149: 
150: ## 🖊️ Citation
151: 
152: If you find this work helpful, please cite us:
153: 
154: ```bibtex
155: @article{2026cliffordnet,
156:   title={CliffordNet: All You Need is Geometric Algebra},
157:   author={Zhongping Ji},
158:   journal={arXiv preprint arXiv:2601.06793},
159:   year={2026}
160: }
161: ```
162: 
163: ## 🙏 Acknowledgement
164: 
165: We thank the open-source community for the implementations of `timm`, which facilitated our baseline comparisons.
````

## File: train.py
````python
  1: import time
  2: import torch
  3: import torch.nn as nn
  4: import torch.optim as optim
  5: import torchvision
  6: import torchvision.transforms as transforms
  7: from torch.utils.data import DataLoader
  8: from tqdm import tqdm
  9: from dataclasses import dataclass
 10: import argparse
 11: from model import CliffordNet
 12: from networks import cliffordnet_12_2, cliffordnet_12_5, cliffordnet_32_3 
 13: from utils import seed_everything
 14: # from hybrid_model import clifford_hybrid_nano
 15: 
 16: # --- Configuration ---
 17: @dataclass
 18: class TrainingConfig:
 19:     batch_size: int = 128
 20:     lr: float = 1e-3
 21:     epochs: int = 200
 22:     weight_decay: float = 0.1
 23:     num_workers: int = 4 if torch.cuda.is_available() else 0
 24:     device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
 25:     
 26:     # Dataset params
 27:     data_root: str = './data'
 28:     random_erasing_prob: float = 0.25
 29:     
 30:     num_classes: int = 100
 31:     patch_size: int = 2
 32:     embed_dim: int = 128
 33:     
 34:     # Checkpoint
 35:     save_path: str = 'cliffordnet_cifar100.pth'
 36: 
 37: # --- Utils ---
 38: def get_device(device_str: str) -> torch.device:
 39:     print(f"Using Device: {device_str.upper()}")
 40:     return torch.device(device_str)
 41: 
 42: def count_parameters(model: nn.Module) -> int:
 43:     return sum(p.numel() for p in model.parameters() if p.requires_grad)
 44: 
 45: # --- Data Pipeline ---
 46: def get_transforms(cfg: TrainingConfig):
 47:     transform_train = transforms.Compose([
 48:         transforms.RandomCrop(32, padding=4),
 49:         transforms.RandomHorizontalFlip(),
 50:         transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
 51:         transforms.ToTensor(),
 52:         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
 53:         transforms.RandomErasing(p=cfg.random_erasing_prob)
 54:     ])
 55:     transform_test = transforms.Compose([
 56:         transforms.ToTensor(),
 57:         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
 58:     ])
 59:     
 60:     return transform_train, transform_test
 61: 
 62: def get_dataloaders(cfg: TrainingConfig):
 63:     print("Preparing Data...")
 64:     train_transform, test_transform = get_transforms(cfg)
 65:     
 66:     trainset = torchvision.datasets.CIFAR100(
 67:         root=cfg.data_root, train=True, download=True, transform=train_transform
 68:     )
 69:     testset = torchvision.datasets.CIFAR100(
 70:         root=cfg.data_root, train=False, download=True, transform=test_transform
 71:     )
 72: 
 73:     trainloader = DataLoader(
 74:         trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
 75:     )
 76:     testloader = DataLoader(
 77:         testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
 78:     )
 79:     
 80:     return trainloader, testloader
 81: 
 82: # --- Training Engine ---
 83: def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
 84:     model.train()
 85:     running_loss = 0.0
 86:     correct = 0
 87:     total = 0
 88:     
 89:     pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)
 90:     
 91:     for inputs, labels in pbar:
 92:         inputs, labels = inputs.to(device), labels.to(device)
 93:         
 94:         optimizer.zero_grad()
 95:         outputs = model(inputs)
 96:         loss = criterion(outputs, labels)
 97:         loss.backward()
 98:         optimizer.step()
 99:         
100:         running_loss += loss.item()
101:         _, predicted = outputs.max(1)
102:         total += labels.size(0)
103:         correct += predicted.eq(labels).sum().item()
104:         
105:         pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")
106: 
107: @torch.no_grad()
108: def evaluate(model, loader, device, epoch, best_acc, save_path='best_model.pth'):
109: 
110:     model.eval()
111:     correct = 0
112:     total = 0
113:     
114:     for inputs, labels in loader:
115:         inputs, labels = inputs.to(device), labels.to(device)
116:         outputs = model(inputs)
117:         _, predicted = outputs.max(1)
118:         total += labels.size(0)
119:         correct += predicted.eq(labels).sum().item()
120:     
121:     acc = 100. * correct / total
122:     print(f"Epoch {epoch} Test Acc: {acc:.2f}%")
123:     
124:     if acc > best_acc:
125:         print(f"🔥 New record! Accuracy improved from {best_acc:.2f}% to {acc:.2f}%")
126:         best_acc = acc
127: 
128:     return best_acc
129: 
130: 
131: # --- Main Execution ---
132: def main(enable_cuda=False):
133: 
134:     # 1. Setup
135:     cfg = TrainingConfig()
136:     seed_everything()
137:     device = get_device(cfg.device)
138:     
139:     # 2. Data
140:     trainloader, testloader = get_dataloaders(cfg)
141:     
142:     # 3. Model Initialization
143:     print("Initializing Model...")
144: 
145:     try:
146:         from networks import cliffordnet_12_2
147:         model = cliffordnet_12_2(
148:             num_classes=cfg.num_classes, 
149:             patch_size=cfg.patch_size, 
150:             embed_dim=cfg.embed_dim,
151:             enable_cuda=enable_cuda
152:         )
153:     except ImportError:
154:         print("Warning: model not found, using generic CliffordNet.")
155:         model = CliffordNet(
156:             num_classes=cfg.num_classes,
157:             img_size=32, 
158:             patch_size=cfg.patch_size, 
159:             embed_dim=cfg.embed_dim,
160:             depth=12, 
161:             enable_cuda=enable_cuda
162:         )
163:       
164:     model = model.to(device)
165:     
166:     print(f"Model built. Learnable Parameters: {count_parameters(model):,}")
167: 
168:     # 4. Optimization Components
169:     criterion = nn.CrossEntropyLoss()
170:     optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
171:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
172: 
173:     # 5. Training Loop
174:     print(f"Start training for {cfg.epochs} epochs...")
175:     start_time = time.time()
176:     best_acc = 0.0 
177:     for epoch in range(1, cfg.epochs + 1):
178:         train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, cfg.epochs)
179:         best_acc = evaluate(model, testloader, device, epoch, best_acc, save_path=cfg.save_path)
180:         scheduler.step()
181: 
182:     total_time = time.time() - start_time
183:     print(f"Training Finished. Total time: {total_time/60:.2f} mins")
184: 
185: 
186:     
187: if __name__ == "__main__":
188:     
189:     parser = argparse.ArgumentParser(description="supports CUDA acceleration")
190:     parser.add_argument('--enable_cuda', action='store_true', help='Whether to enable CUDA acceleration (default: False)')
191:     args = parser.parse_args()    
192:     print(f"Enable CUDA acceleration: {args.enable_cuda}")
193:     main(enable_cuda=args.enable_cuda)
````

## File: utils.py
````python
 1: import torch
 2: import torch.nn as nn
 3: import numpy as np
 4: import random
 5: import os
 6: 
 7: 
 8: def seed_everything(seed=42):
 9:     random.seed(seed)
10:     os.environ['PYTHONHASHSEED'] = str(seed)
11:     np.random.seed(seed)
12:     torch.manual_seed(seed)
13:     if torch.cuda.is_available():
14:         torch.cuda.manual_seed(seed)
15:         torch.cuda.manual_seed_all(seed)
16:         torch.backends.cudnn.deterministic = True
17:         torch.backends.cudnn.benchmark = False
18:     elif torch.backends.mps.is_available():
19:         torch.mps.manual_seed(seed)
20:     print(f"Global seed set to {seed}")
21: 
22: def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
23:     if drop_prob == 0. or not training:
24:         return x
25:     keep_prob = 1 - drop_prob
26:     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
27:     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
28:     if keep_prob > 0.0 and scale_by_keep:
29:         random_tensor.div_(keep_prob)
30:     return x * random_tensor
31: 
32: class DropPath(nn.Module):
33:     def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
34:         super(DropPath, self).__init__()
35:         self.drop_prob = drop_prob
36:         self.scale_by_keep = scale_by_keep
37: 
38:     def forward(self, x):
39:         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
````
