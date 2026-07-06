"""ULTRAFOOT シリアルトラフィック キャプチャ プロキシ

ULTRAFOOT がセンサーに送信するコマンドを記録する。
com0com の仮想ポートペアを使って ULTRAFOOT と実センサーの間に割り込む。

== セットアップ手順 ==

1. com0com をインストール
   https://sourceforge.net/projects/com0com/

2. com0com セットアップで仮想ポートペアを作成
   例: COM10 <-> COM11

3. ULTRAFOOT の Scanner.dat を編集
   元: com5  →  変更後: com10
   （Scanner.dat は ULTRAFOOT.exe と同じフォルダにある）

4. このスクリプトを起動（ULTRAFOOT より先に起動すること）
   python sensor_proxy.py --ultrafoot COM11 --sensor COM5

5. ULTRAFOOT を起動して通常操作を行う
   「スキャン」「リセット」ボタンを押して動作を確認

6. ログファイル (proxy_log.txt) に記録されたコマンドを確認
   ULTRAFOOT→センサー の行に注目する

7. 作業終了後: Scanner.dat を com5 に戻す

== ログの見方 ==
  TX> 53                 … ULTRAFOOT が b'S' (0x53) を送信（スキャンコマンド）
  RX< 0300 0304 ...     … センサーからの応答
"""
import argparse
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "proxy_log.txt"


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def hex_bytes(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)


def relay(src, dst, direction: str, log_file) -> None:
    """src から読んで dst に転送しログに記録する（ブロッキング・スレッド用）"""
    prefix = "TX>" if direction == "tx" else "RX<"
    try:
        while True:
            # 最大256バイトずつ読む
            data = src.read(256)
            if not data:
                time.sleep(0.005)
                continue
            dst.write(data)
            line = f"[{timestamp()}] {prefix} {hex_bytes(data)}\n"
            print(line, end="")
            log_file.write(line)
            log_file.flush()

            # テキスト形式で読める場合はデコードも表示
            try:
                decoded = data.decode("ascii").strip()
                if decoded:
                    note = f"           ({repr(decoded)})\n"
                    print(note, end="")
                    log_file.write(note)
                    log_file.flush()
            except Exception:
                pass
    except Exception as exc:
        print(f"[{direction}] リレー終了: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ULTRAFOOT シリアルプロキシ")
    parser.add_argument("--ultrafoot", default="COM11",
                        help="ULTRAFOOT が接続する仮想ポート (com0com ペアの片方)")
    parser.add_argument("--sensor", default="COM5",
                        help="実センサーの物理ポート")
    parser.add_argument("--baud", type=int, default=19200, help="ボーレート (default: 19200)")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial が未インストール: pip install pyserial")
        sys.exit(1)

    print(f"センサープロキシ起動")
    print(f"  ULTRAFOOT 側ポート : {args.ultrafoot}")
    print(f"  実センサー側ポート : {args.sensor}")
    print(f"  ボーレート         : {args.baud}")
    print(f"  ログファイル       : {LOG_FILE}")
    print()
    print("ULTRAFOOT を起動して操作してください。Ctrl+C で終了。\n")

    try:
        ser_ultrafoot = serial.Serial(args.ultrafoot, args.baud, timeout=0)
    except Exception as exc:
        print(f"[ERROR] {args.ultrafoot} を開けません: {exc}")
        sys.exit(1)

    try:
        ser_sensor = serial.Serial(args.sensor, args.baud, timeout=0)
    except Exception as exc:
        print(f"[ERROR] {args.sensor} を開けません: {exc}")
        ser_ultrafoot.close()
        sys.exit(1)

    with open(LOG_FILE, "w", encoding="utf-8") as lf:
        lf.write(f"=== proxy 開始: {datetime.now()} ===\n")
        lf.write(f"ULTRAFOOT={args.ultrafoot}  SENSOR={args.sensor}  BAUD={args.baud}\n\n")

        t_tx = threading.Thread(
            target=relay, args=(ser_ultrafoot, ser_sensor, "tx", lf), daemon=True
        )
        t_rx = threading.Thread(
            target=relay, args=(ser_sensor, ser_ultrafoot, "rx", lf), daemon=True
        )
        t_tx.start()
        t_rx.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nプロキシを停止します...")

        lf.write(f"\n=== proxy 終了: {datetime.now()} ===\n")

    ser_ultrafoot.close()
    ser_sensor.close()
    print(f"ログを保存しました: {LOG_FILE}")
    print("\n=== TX> 行の内容（ULTRAFOOT→センサーのコマンド一覧）===")
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        if "TX>" in line:
            print(line)


if __name__ == "__main__":
    main()
