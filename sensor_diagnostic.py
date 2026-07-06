"""センサー診断スクリプト — 現在の ADC 値を確認して状態を判定する

安全: b'S' コマンドのみ送信（読み取り専用）

使い方:
    python sensor_diagnostic.py [--port COM5] [--count 3]

ULTRAFOOT を閉じてから実行すること。
"""
import argparse
import sys
import time
import numpy as np

SENSOR_ROWS = 32
SENSOR_COLS = 16
RESPONSE_BYTES = SENSOR_ROWS * SENSOR_COLS * 2 + 2  # 1026 bytes
BAUD = 19200
SCAN_CMD = b"S"

# 正常値の目安（空状態）
EXPECTED_EMPTY_MIN = 700
EXPECTED_EMPTY_MAX = 950


def scan_once(ser) -> np.ndarray | None:
    ser.reset_input_buffer()
    ser.write(SCAN_CMD)

    buf = bytearray()
    deadline = time.time() + 3.0
    while time.time() < deadline:
        n = ser.in_waiting
        if n > 0:
            buf.extend(ser.read(n))
            deadline = time.time() + 0.3
        else:
            time.sleep(0.05)
        if len(buf) >= RESPONSE_BYTES:
            break

    if len(buf) < SENSOR_ROWS * SENSOR_COLS * 2:
        print(f"[ERROR] 受信バイト数不足: {len(buf)} / {SENSOR_ROWS * SENSOR_COLS * 2}")
        return None

    values = np.frombuffer(
        bytes(buf[:SENSOR_ROWS * SENSOR_COLS * 2]),
        dtype=">u2",
    ).astype(np.float32)
    return values.reshape(SENSOR_ROWS, SENSOR_COLS)


def print_grid(grid: np.ndarray) -> None:
    print("\n  ADC グリッド (32行 × 16列)  低い値=接触あり")
    print("  col: " + "  ".join(f"{c:3d}" for c in range(SENSOR_COLS)))
    print("  " + "-" * (SENSOR_COLS * 5 + 2))
    for r in range(SENSOR_ROWS):
        row = grid[r]
        row_str = "  ".join(f"{int(v):3d}" for v in row)
        print(f"  row{r:02d}: {row_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="センサー ADC 診断")
    parser.add_argument("--port", default="COM5", help="シリアルポート (default: COM5)")
    parser.add_argument("--count", type=int, default=3, help="スキャン回数 (default: 3)")
    parser.add_argument("--grid", action="store_true", help="全セルの ADC 値をグリッド表示")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial が未インストール: pip install pyserial")
        sys.exit(1)

    print(f"センサー診断を開始します  ポート={args.port}  回数={args.count}")
    print("センサーは何も乗せていない状態（空）で実行してください。\n")

    try:
        ser = serial.Serial(args.port, BAUD, timeout=1)
    except Exception as exc:
        print(f"[ERROR] ポートを開けません: {exc}")
        print("ULTRAFOOT が起動している場合は終了してから実行してください。")
        sys.exit(1)

    time.sleep(0.3)
    ser.reset_input_buffer()

    all_means: list[float] = []

    for i in range(args.count):
        print(f"--- スキャン {i + 1}/{args.count} ---")
        grid = scan_once(ser)
        if grid is None:
            continue

        mn = float(grid.min())
        mx = float(grid.max())
        mean = float(grid.mean())
        all_means.append(mean)

        print(f"  min={mn:.0f}  max={mx:.0f}  mean={mean:.1f}")

        if args.grid:
            print_grid(grid)

        time.sleep(0.5)

    ser.close()

    if not all_means:
        print("\n[FAIL] 有効なスキャンが取得できませんでした。")
        sys.exit(1)

    avg = sum(all_means) / len(all_means)
    print(f"\n===== 診断結果 =====")
    print(f"平均 ADC (mean): {avg:.1f}")
    print(f"正常値目安: {EXPECTED_EMPTY_MIN} ～ {EXPECTED_EMPTY_MAX}")

    if EXPECTED_EMPTY_MIN <= avg <= EXPECTED_EMPTY_MAX:
        print("→ [OK] センサーの ADC 値は正常範囲内です。")
        print("   ULTRAFOOT での表示異常はソフト側の問題の可能性があります。")
    elif avg < EXPECTED_EMPTY_MIN:
        print(f"→ [NG] ADC 値が低すぎます（{avg:.0f} < {EXPECTED_EMPTY_MIN}）。")
        print("   センサーの0点がズレている可能性があります。sensor_proxy.py でコマンドをキャプチャしてください。")
    else:
        print(f"→ [NG] ADC 値が高すぎます（{avg:.0f} > {EXPECTED_EMPTY_MAX}）。")
        print("   センサーの状態を確認してください。")


if __name__ == "__main__":
    main()
