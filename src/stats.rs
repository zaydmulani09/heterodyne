//! Exact full-vocabulary comparison of two next-token distributions, one row per position.
//!
//! All sums run in f64 straight from the raw f32 logits:
//!   KL(P_a || P_b) = sum_v p_a(v) * (a_v - b_v) - lse(a) + lse(b)
//! so KL(A || A) is exactly 0 and no intermediate log-prob rounding adds a noise floor.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Conventions, not tests: a top-1 flip is a near-tie when, under arm A, B's choice was within
/// this many nats of A's choice, and confident when A preferred its own choice by at least
/// `CONFIDENT` nats.
pub const NEAR_TIE: f64 = 0.25;
pub const CONFIDENT: f64 = 1.0;

fn lse_argmax2(row: &[f32]) -> (f64, [usize; 2]) {
    let (mut i1, mut i2) = (0usize, usize::MAX);
    for (i, &x) in row.iter().enumerate().skip(1) {
        if x > row[i1] {
            i2 = i1;
            i1 = i;
        } else if i2 == usize::MAX || x > row[i2] {
            i2 = i;
        }
    }
    let m = row[i1] as f64;
    let s: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
    (m + s.ln(), [i1, if i2 == usize::MAX { i1 } else { i2 }])
}

fn kl_row(a: &[f32], b: &[f32], lse_a: f64, lse_b: f64) -> f64 {
    let mut acc = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = x as f64 - y as f64;
        if d != 0.0 {
            acc += (x as f64 - lse_a).exp() * d;
        }
    }
    // Clamp tiny negative values from rounding; KL is non-negative.
    (acc - lse_a + lse_b).max(0.0)
}

/// KL(P_a || P_b) for every row; `a`, `b` are row-major `[n][v]` logits.
pub fn kl_rows(a: &[f32], b: &[f32], v: usize) -> Vec<f64> {
    a.par_chunks(v)
        .zip(b.par_chunks(v))
        .map(|(ra, rb)| {
            if ra == rb {
                return 0.0;
            }
            kl_row(ra, rb, lse_argmax2(ra).0, lse_argmax2(rb).0)
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PosStat {
    pub kl: f64,
    pub top_a: i32,
    pub top_b: i32,
    /// log p_a(top1) - log p_a(top2): how decided arm A was.
    pub margin_a: f64,
    /// log p_a(A's top) - log p_a(B's top): how much A preferred its choice over B's (0 if same).
    #[serde(default)]
    pub gap_ab: f64,
    pub entropy_a: f64,
    /// log-prob of the actual next token (absent on the last position).
    pub lp_next_a: Option<f64>,
    pub lp_next_b: Option<f64>,
    /// Top-k (token, log-prob) under each arm.
    pub topk_a: Vec<(i32, f32)>,
    pub topk_b: Vec<(i32, f32)>,
}

fn topk(row: &[f32], lse: f64, k: usize) -> Vec<(i32, f32)> {
    if k == 0 {
        return vec![];
    }
    let mut idx: Vec<usize> = Vec::with_capacity(k + 1);
    for (i, &x) in row.iter().enumerate() {
        if idx.len() < k || x > row[idx[idx.len() - 1]] {
            let at = idx.partition_point(|&j| row[j] >= x);
            idx.insert(at, i);
            idx.truncate(k);
        }
    }
    idx.into_iter().map(|i| (i as i32, (row[i] as f64 - lse) as f32)).collect()
}

/// Full per-position comparison. `next[i]` is the token at position i+1 (the target).
pub fn compare(a: &[f32], b: &[f32], v: usize, next: &[i32], k: usize) -> Vec<PosStat> {
    a.par_chunks(v)
        .zip(b.par_chunks(v))
        .enumerate()
        .map(|(i, (ra, rb))| {
            let (lse_a, ta) = lse_argmax2(ra);
            let (lse_b, tb) = lse_argmax2(rb);
            let entropy_a = -ra
                .iter()
                .map(|&x| {
                    let lp = x as f64 - lse_a;
                    lp.exp() * lp
                })
                .sum::<f64>();
            let tgt = next.get(i).map(|&t| t as usize);
            PosStat {
                kl: if ra == rb { 0.0 } else { kl_row(ra, rb, lse_a, lse_b) },
                top_a: ta[0] as i32,
                top_b: tb[0] as i32,
                margin_a: ra[ta[0]] as f64 - ra[ta[1]] as f64,
                gap_ab: ra[ta[0]] as f64 - ra[tb[0]] as f64,
                entropy_a: entropy_a.max(0.0),
                lp_next_a: tgt.map(|t| ra[t] as f64 - lse_a),
                lp_next_b: tgt.map(|t| rb[t] as f64 - lse_b),
                topk_a: topk(ra, lse_a, k),
                topk_b: topk(rb, lse_b, k),
            }
        })
        .collect()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub n_positions: usize,
    pub mean_kl: f64,
    /// 95% t-interval for the mean, treating chunks as clusters (absent with one chunk).
    /// With few chunks it is wide and only approximate; lower bound clamped at 0.
    pub mean_kl_ci95: Option<[f64; 2]>,
    #[serde(default)]
    pub n_chunks: usize,
    pub median_kl: f64,
    pub p99_kl: f64,
    pub max_kl: f64,
    pub flips: usize,
    pub flips_near_tie: usize,
    pub flips_confident: usize,
    pub flip_rate: f64,
    /// exp(mean NLL of the actual next tokens) under each arm.
    pub ppl_a: Option<f64>,
    pub ppl_b: Option<f64>,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

pub fn summarize(chunks: &[&[PosStat]]) -> Summary {
    let all: Vec<&PosStat> = chunks.iter().flat_map(|c| c.iter()).collect();
    let n = all.len();
    if n == 0 {
        return Summary::default();
    }
    let mut kls: Vec<f64> = all.iter().map(|p| p.kl).collect();
    kls.sort_by(|x, y| x.total_cmp(y));
    let flipped: Vec<&&PosStat> = all.iter().filter(|p| p.top_a != p.top_b).collect();
    let nll = |f: fn(&PosStat) -> Option<f64>| {
        let v: Vec<f64> = all.iter().filter_map(|p| f(p)).collect();
        (!v.is_empty()).then(|| (-v.iter().sum::<f64>() / v.len() as f64).exp())
    };
    let chunk_sums: Vec<(f64, usize)> = chunks.iter().map(|c| (c.iter().map(|p| p.kl).sum(), c.len())).collect();
    Summary {
        n_positions: n,
        mean_kl: kls.iter().sum::<f64>() / n as f64,
        mean_kl_ci95: cluster_t_ci(&chunk_sums),
        n_chunks: chunks.len(),
        median_kl: quantile(&kls, 0.5),
        p99_kl: quantile(&kls, 0.99),
        max_kl: *kls.last().unwrap(),
        flips: flipped.len(),
        flips_near_tie: flipped.iter().filter(|p| p.gap_ab < NEAR_TIE).count(),
        flips_confident: flipped.iter().filter(|p| p.gap_ab >= CONFIDENT).count(),
        flip_rate: flipped.len() as f64 / n as f64,
        ppl_a: nll(|p| p.lp_next_a),
        ppl_b: nll(|p| p.lp_next_b),
    }
}

/// Two-sided 95% Student t quantile for `df` degrees of freedom (table + linear interpolation).
fn t975(df: usize) -> f64 {
    const T: [(usize, f64); 25] = [
        (1, 12.706),
        (2, 4.303),
        (3, 3.182),
        (4, 2.776),
        (5, 2.571),
        (6, 2.447),
        (7, 2.365),
        (8, 2.306),
        (9, 2.262),
        (10, 2.228),
        (11, 2.201),
        (12, 2.179),
        (13, 2.160),
        (14, 2.145),
        (15, 2.131),
        (16, 2.120),
        (17, 2.110),
        (18, 2.101),
        (19, 2.093),
        (20, 2.086),
        (30, 2.042),
        (40, 2.021),
        (60, 2.000),
        (120, 1.980),
        (100_000, 1.960),
    ];
    let df = df.max(1);
    let i = T.partition_point(|&(d, _)| d < df).min(T.len() - 1);
    if T[i].0 == df || i == 0 {
        return T[i].1;
    }
    let ((d0, t0), (d1, t1)) = (T[i - 1], T[i]);
    t0 + (t1 - t0) * (df - d0) as f64 / (d1 - d0) as f64
}

/// 95% interval for a position-weighted mean when positions are clustered in chunks
/// (positions within a chunk are correlated). Cluster-robust standard error,
/// t quantile with k-1 degrees of freedom; equal-size chunks reduce to the classic
/// t-interval over chunk means. Input: (sum of KL, positions) per chunk.
pub fn cluster_t_ci(chunks: &[(f64, usize)]) -> Option<[f64; 2]> {
    let k = chunks.len();
    if k < 2 {
        return None;
    }
    let total: f64 = chunks.iter().map(|c| c.1 as f64).sum();
    let mean = chunks.iter().map(|c| c.0).sum::<f64>() / total;
    let ss: f64 = chunks.iter().map(|&(s, n)| (s - n as f64 * mean).powi(2)).sum();
    let se = (k as f64 / (k - 1) as f64 * ss).sqrt() / total;
    let h = t975(k - 1) * se;
    Some([(mean - h).max(0.0), mean + h])
}

/// ||h_b - h_a|| / ||h_a|| per row of width `w` (residual-stream relative error).
pub fn rel_err_rows(a: &[f32], b: &[f32], w: usize) -> Vec<f32> {
    a.chunks(w)
        .zip(b.chunks(w))
        .map(|(ra, rb)| {
            let (mut d, mut n) = (0f64, 0f64);
            for (&x, &y) in ra.iter().zip(rb) {
                d += (y as f64 - x as f64).powi(2);
                n += (x as f64).powi(2);
            }
            (d.sqrt() / n.sqrt().max(1e-30)) as f32
        })
        .collect()
}
