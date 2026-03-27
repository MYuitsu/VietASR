# VietASR Rust Mic Web App

Web app Rust nhỏ để test microphone trên trình duyệt và gửi audio vào checkpoint VietASR hiện có.

## Chạy

1. Cài Rust toolchain nếu máy chưa có `cargo`.
2. Từ repo root:

```bash
cd /home/nguyenthaiduy277/VietASR-1/apps/mic_web_rust

export VIETASR_REPO_ROOT=/home/nguyenthaiduy277/VietASR-1
export VIETASR_CHECKPOINT=/home/nguyenthaiduy277/VietASR-1/hf_models/viet_iter3_pseudo_label/exp/epoch-12.pt
export VIETASR_TOKENS=/home/nguyenthaiduy277/VietASR-1/hf_models/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt

cargo run
```

Mở `http://127.0.0.1:3000`.

## Yêu cầu

- `python3`
- `ffmpeg`
- dependency Python của VietASR đã cài như workspace hiện tại
- checkpoint và `tokens.txt`

## Ghi chú

Backend Rust gọi helper Python ở `apps/mic_web_rust/transcribe_once.py` để tận dụng model ASR sẵn có của repo.
