import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title='Parallel Computing Lab')

class Experiment(BaseModel):
    limit: int = Field(default=200000, ge=1000, le=3000000)
    workers: list[int] = Field(default=[1, 2, 4, 8], min_length=1, max_length=8)
    repeats: int = Field(default=3, ge=1, le=5)

def count_primes(bounds: tuple[int, int]) -> int:
    start, end = bounds
    count = 0
    for n in range(max(2, start), end):
        if n == 2:
            count += 1
        elif n % 2 and all(n % d for d in range(3, math.isqrt(n) + 1, 2)):
            count += 1
    return count

def chunks(limit: int, workers: int) -> list[tuple[int, int]]:
    size = math.ceil((limit - 1) / workers)
    return [(2 + i * size, min(limit + 1, 2 + (i + 1) * size)) for i in range(workers) if 2 + i * size <= limit]

def measure(limit: int, workers: int) -> tuple[float, int]:
    ranges = chunks(limit, workers)
    started = time.perf_counter()
    if workers == 1:
        total = count_primes(ranges[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            total = sum(executor.map(count_primes, ranges))
    return time.perf_counter() - started, total

@app.get('/')
def home():
    return FileResponse('static/index.html')

@app.post('/api/benchmark')
def benchmark(exp: Experiment):
    workers = sorted(set(exp.workers) | {1})
    if any(w < 1 or w > 16 for w in workers):
        raise HTTPException(422, 'Workers must be between 1 and 16')
    results = []
    expected = None
    for w in workers:
        samples = []
        for _ in range(exp.repeats):
            elapsed, total = measure(exp.limit, w)
            if expected is None:
                expected = total
            elif total != expected:
                raise HTTPException(500, 'Inconsistent prime counts')
            samples.append(elapsed)
        results.append({'workers': w, 'seconds': round(statistics.median(samples), 5), 'samples': [round(x, 5) for x in samples]})
    baseline = next(r['seconds'] for r in results if r['workers'] == 1)
    for r in results:
        r['speedup'] = round(baseline / r['seconds'], 3) if r['seconds'] else 0
        r['efficiency'] = round(100 * r['speedup'] / r['workers'], 2)
    return {'limit': exp.limit, 'prime_count': expected, 'repeats': exp.repeats, 'cpu_count': os.cpu_count(), 'results': results}
