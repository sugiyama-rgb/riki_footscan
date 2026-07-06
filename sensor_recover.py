"""センサー回復コマンド送信スクリプト

sensor_proxy.py でキャプチャしたコマンドをここに設定して実行する。

使い方:
    # proxy_log.txt の TX> 行を確認してコマンドを特定したら:
    python sensor_recover.py --cmd "ScanE\r\n"  # 例（実際のコマンドに差し替える）

    # バイト列で直接指定する場合:
    python sensor_recover.py --hex "5363616e450d0a"

実行後、sensor_diagnostic.py で ADC 値が正常に戻ったか確認すること。
"""
import argparse
import sys
import time
import numpy as np

BAUD = 19200
SENSOR_ROWS = 32
SENSOR_COLS = 16
SCAN_CMD = b"S"


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
        if len(buf) >= SENSOR_ROWS * SENSOR_COLS * 2:
            break
    if len(buf) < SENSOR_ROWS * SENSOR_COLS * 2:
        return None
    values = np.frombuffer(
        bytes(buf[:SENSOR_ROWS * SENSOR_COLS * 2]), dtype=">u2"
    ).astype(np.float32)
    return values.reshape(SENSOR_ROWS, SENSOR_COLS)


def adc_summary(label: str, grid: np.ndarray) -> None:
    print(f"  {label}: min={grid.min():.0f}  max={grid.max():.0f}  mean={grid.mean():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="センサー回復コマンド送信")
    parser.add_argument("--port", default="COM5", help="シリアルポート (default: COM5)")
    parser.add_argument(
        "--cmd", default=None,
        help="送信するテキストコマンド（エスケープ有効、例: 'ScanE\\r\\n'）"
    )
    parser.add_argument(
        "--hex", default=None,
        help="送信するバイト列を16進数で指定（例: '5363616e450d0a'）"
    )
    parser.add_argument(
        "--response-timeout", type=float, default=2.0,
        help="コマンド送信後の応答待ち時間（秒）(default: 2.0)"
    )
    args = parser.parse_args()

    # コマンドのバイト列を構築
    if args.hex:
        try:
            cmd_bytes = bytes.fromhex(args.hex)
        except ValueError as exc:
            print(f"[ERROR] --hex の形式が無効: {exc}")
            sys.exit(1)
    elif args.cmd:
        cmd_bytes = args.cmd.encode("ascii").decode("unicode_escape").encode("ascii")
    else:
        print("[ERROR] --cmd または --hex でコマンドを指定してください。")
        print()
        print("sensor_proxy.py のログ (proxy_log.txt) を参照して")
        print("ULTRAFOOT がリセット時に送信したコマンドを確認してください。")
        sys.exit(1)

    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial が未インストール: pip install pyserial")
        sys.exit(1)

    print(f"回復コマンド送信  ポート={args.port}")
    print(f"送信コマンド: {repr(cmd_bytes)}")
    print()

    try:
        ser = serial.Serial(args.port, BAUD, timeout=1)
    except Exception as exc:
        print(f"[ERROR] ポートを開けません: {exc}")
        sys.exit(1)

    time.sleep(0.3)
    ser.reset_input_buffer()

    # ---- Step 1: コマンド送信前の ADC 値を記録 ----
    print("=== コマンド送信前 ===")
    grid_before = scan_once(ser)
    if grid_before is not None:
        adc_summary("ADC", grid_before)
    else:
        print("  [WARN] スキャン失敗（前）")

    time.sleep(0.5)

    # ---- Step 2: 回復コマンドを送信 ----
    print(f"\n=== コマンド送信: {repr(cmd_bytes)} ===")
    ser.reset_input_buffer()
    ser.write(cmd_bytes)

    # 応答を読み取る
    time.sleep(0.1)
    deadline = time.time() + args.response_timeout
    response = bytearray()
    while time.time() < deadline:
        n = ser.in_waiting
        if n > 0:
            response.extend(ser.read(n))
            deadline = time.time() + 0.3
        else:
            time.sleep(0.05)

    if response:
        print(f"応答 ({len(response)} bytes): {response.hex()}")
        try:
            print(f"テキスト応答: {repr(response.decode('ascii'))}")
        except Exception:
            pass
    else:
        print("応答なし（タイムアウト）")

    time.sleep(1.0)

    # ---- Step 3: コマンド送信後の ADC 値を確認 ----
    print("\n=== コマンド送信後 ===")
    grid_after = scan_once(ser)
    if grid_after is not None:
        adc_summary("ADC", grid_after)
    else:
        print("  [WARN] スキャン失敗（後）")

    ser.close()

    # ---- 結果比較 ----
    print("\n=== 変化の確認 ===")
    if grid_before is not None and grid_after is not None:
        diff = float(grid_after.mean()) - float(grid_before.mean())
        print(f"平均 ADC の変化: {diff:+.1f}")
        if abs(diff) < 5:
            print("→ ADC 値にほとんど変化なし（コマンドが効いていない可能性）")
        else:
            print(f"→ ADC 値が変化しました（回復の兆候かもしれません）")
    print("\nsensor_diagnostic.py で詳細な状態を確認してください。")


if __name__ == "__main__":
    main()
