use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::Path;

/// GPT-2 config parsed from HuggingFace config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

impl Gpt2Config {
    pub fn from_json(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// Load a Conv1D-style weight: GPT-2 stores [in, out], candle Linear expects [out, in].
fn linear_conv1d(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get((in_d, out_d), "weight")?.t()?;
    let b = vb.get(out_d, "bias")?;
    Ok(Linear::new(w, Some(b)))
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn make_causal_mask(seq_len: usize, past_len: usize, device: &Device) -> Result<Tensor> {
    let total_len = past_len + seq_len;
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                if j <= past_len + i {
                    1.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, seq_len, total_len), device)
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let n_embd = cfg.n_embd;
        let c_attn = linear_conv1d(n_embd, 3 * n_embd, vb.pp("c_attn"))?;
        let c_proj = linear_conv1d(n_embd, n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head: cfg.n_head,
            head_dim: n_embd / cfg.n_head,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;

        let q = qkv.narrow(D::Minus1, 0, c)?;
        let k = qkv.narrow(D::Minus1, c, c)?;
        let v = qkv.narrow(D::Minus1, 2 * c, c)?;

        // Reshape to [B, n_head, T, head_dim]
        let q = q
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? / scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        // [B, n_head, T, head_dim] -> [B, T, C]
        let out = out.transpose(1, 2)?.reshape((b, t, c))?;
        self.c_proj.forward(&out)
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let n_embd = cfg.n_embd;
        let inner_dim = 4 * n_embd;
        let c_fc = linear_conv1d(n_embd, inner_dim, vb.pp("c_fc"))?;
        let c_proj = linear_conv1d(inner_dim, n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c_fc.forward(x)?.gelu()?;
        self.c_proj.forward(&h)
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Gpt2Config) -> Result<Self> {
        let ln_1 = layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::load(vb.pp("attn"), cfg)?;
        let ln_2 = layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Pre-norm residual
        let h = self.ln_1.forward(x)?;
        let h = self.attn.forward(&h, mask)?;
        let x = (x + h)?;
        let h = self.ln_2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        &x + h
    }
}

/// A shard of GPT-2 containing a subset of layers.
///
/// - First shard has wte + wpe embeddings.
/// - Last shard has ln_f and uses tied wte.weight as lm_head.
pub struct Gpt2Shard {
    wte: Option<Embedding>,
    wpe: Option<Embedding>,
    blocks: Vec<Block>,
    ln_f: Option<LayerNorm>,
    lm_head_weight: Option<Tensor>,
    #[allow(dead_code)]
    cfg: Gpt2Config,
    is_first: bool,
    is_last: bool,
}

impl Gpt2Shard {
    /// Load a shard covering layers [layer_start, layer_end).
    /// `is_first` loads wte/wpe, `is_last` loads ln_f and ties lm_head to wte.
    pub fn load(
        model_dir: &Path,
        cfg: &Gpt2Config,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[safetensors_path],
                DType::F32,
                device,
            )?
        };

        let prefix = detect_prefix(model_dir)?;
        let vb_model = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(&prefix)
        };

        let wte = if is_first {
            Some(candle_nn::embedding(
                cfg.vocab_size,
                cfg.n_embd,
                vb_model.pp("wte"),
            )?)
        } else {
            None
        };

        let wpe = if is_first {
            Some(candle_nn::embedding(
                cfg.n_positions,
                cfg.n_embd,
                vb_model.pp("wpe"),
            )?)
        } else {
            None
        };

        let blocks = (layer_start..layer_end)
            .map(|i| Block::load(vb_model.pp(format!("h.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        let ln_f = if is_last {
            Some(layer_norm(
                cfg.n_embd,
                cfg.layer_norm_epsilon,
                vb_model.pp("ln_f"),
            )?)
        } else {
            None
        };

        // GPT-2 ties lm_head to wte.weight (transposed)
        let lm_head_weight = if is_last {
            let w = vb_model.get((cfg.vocab_size, cfg.n_embd), "wte.weight")?;
            Some(w)
        } else {
            None
        };

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head_weight,
            cfg: cfg.clone(),
            is_first,
            is_last,
        })
    }

    /// Forward pass through this shard.
    ///
    /// - First shard: input_ids [B, T] (u32) → hidden [B, T, n_embd] (f32)
    /// - Middle shard: hidden [B, T, n_embd] → hidden [B, T, n_embd]
    /// - Last shard: hidden [B, T, n_embd] → logits [B, vocab_size] (last token only)
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let device = input.device();

        let mut hidden = if self.is_first {
            // input is [B, T] u32 token ids
            let (b, t) = input.dims2()?;
            let wte = self.wte.as_ref().unwrap();
            let wpe = self.wpe.as_ref().unwrap();

            let position_ids = Tensor::arange(0u32, t as u32, device)?
                .unsqueeze(0)?
                .broadcast_as((b, t))?;
            let token_emb = wte.forward(input)?;
            let pos_emb = wpe.forward(&position_ids)?;
            (&token_emb + &pos_emb)?
        } else {
            // input is already hidden states [B, T, n_embd]
            input.clone()
        };

        let seq_len = hidden.dim(1)?;
        let mask = make_causal_mask(seq_len, 0, device)?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &mask)?;
        }

        if self.is_last {
            let ln_f = self.ln_f.as_ref().unwrap();
            let lm_head_w = self.lm_head_weight.as_ref().unwrap();
            hidden = ln_f.forward(&hidden)?;
            // Take last token only: [B, T, n_embd] -> [B, n_embd]
            let (_b, t, _) = hidden.dims3()?;
            hidden = hidden.narrow(1, t - 1, 1)?.squeeze(1)?;
            // [B, n_embd] @ [n_embd, vocab] = [B, vocab]
            let logits = hidden.matmul(&lm_head_w.t()?)?;
            Ok(logits)
        } else {
            Ok(hidden)
        }
    }
}

/// Detect whether safetensors weights use a "transformer." prefix.
/// Reads the safetensors header to check actual tensor names.
fn detect_prefix(model_dir: &Path) -> anyhow::Result<String> {
    let st_path = model_dir.join("model.safetensors");
    let data = std::fs::read(&st_path)?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    for name in st.names() {
        if name.starts_with("transformer.") {
            return Ok("transformer".to_string());
        }
        if name.starts_with("h.") || name == "wte.weight" || name == "wpe.weight" {
            return Ok(String::new());
        }
    }
    // Default: HF GPT-2 uses "transformer." prefix
    Ok("transformer".to_string())
}
