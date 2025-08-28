import argparse
import statistics
import time
from typing import Dict, List, Optional
from moderators.auto_model import AutoModerator


def summarize(times: List[float]) -> Dict[str, float]:
    times_sorted = sorted(times)
    return {
        "avg_ms": statistics.mean(times) * 1000.0,
        "p50_ms": times_sorted[int(0.5 * (len(times_sorted) - 1))] * 1000.0,
        "p90_ms": times_sorted[int(0.9 * (len(times_sorted) - 1))] * 1000.0,
    }


def benchmark(model_id: str, image_path: str, warmup: int = 2, repeats: int = 10, backend: Optional[str] = None) -> None:
    kwargs = {"backend": backend} if backend else {}
    model = AutoModerator.from_pretrained(model_id, **kwargs)
    # warmup
    for _ in range(warmup):
        model(image_path)

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(image_path)
        times.append(time.perf_counter() - t0)

    stats = summarize(times)
    print(f"Model: {model_id}")
    print(f"Backend: {backend or 'auto'}")
    print(f"Runs: {repeats}, avg: {stats['avg_ms']:.2f} ms, p50: {stats['p50_ms']:.2f} ms, p90: {stats['p90_ms']:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moderators Benchmark")
    parser.add_argument("model_id", help="Hub model id")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backend", default=None, help="Preferred backend (e.g., onnx)")
    args = parser.parse_args()
    benchmark(args.model_id, args.image, args.warmup, args.repeats, args.backend)


if __name__ == "__main__":
    main()

