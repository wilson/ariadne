# Ariadne

**A deterministic measurement engine for jagged, noisy, and irregular data.**

Ariadne is a Zig library for computing the **Path Signature** of a stream of data. It allows you to treat noisy signals (financial ticks, sensor jitters, biological time-series) as geometric objects, extracting robust features without needing to smooth, filter, or model the noise distribution.

## The Problem: Calculus Expects Smoothness

Standard engineering math (Calculus) assumes the world is smooth. It assumes that if you zoom in enough, every curve looks like a straight line with a defined "slope" (derivative).

Real data (stock prices, brownian motion, high-frequency sensor logs) is **jagged**. If you zoom in, it looks like fractals.

* **Differentiation fails:** The "slope" of noise is infinite or undefined.
* **Filtering destroys information:** Smoothing a signal removes the high-frequency interactions that often drive the system (e.g., a sudden spike that triggers a circuit breaker).
* **Sampling matters:** If you sample a loop `A -> B -> A` too slowly, you might just see `A -> A` and assume nothing happened.

## The Solution: The Rough Path Signature

Instead of trying to calculate a *rate of change* (which fails on noise), Ariadne calculates the **Signature** of the path.

Think of the Signature as a **Geometric Hash** or a **Lossless Compression** of the trajectory. It captures not just *where* the path went (displacement), but *how* it got there (area and volume).

### Why Engineers Care

1. **Order Matters (Non-Commutativity):**
* **Scenario:** A robot arm moves `Right`, then `Up`, then `Left`, then `Down`.
* **Net Displacement:** Zero. (It's back where it started).
* **Standard Vector Math:** Sees "Zero Change."
* **Ariadne:** Sees a non-zero **Area** term in the signature. It knows the robot traced a closed loop and may have exerted torque or wound a cable.

2. **Sampling Invariance:**
* Whether you sample a curve with 10 points or 10,000 points, the Signature converges to the same value. You don't need to resample your data to a fixed grid (e.g., "every 1ms") to use it. It eats irregular time-series for breakfast.

3. **"Model-Free" Prediction:**
* You don't need to know if your noise is Gaussian, Lévy, or chaotic. The Signature is just a faithful witness of the geometry. It works on *any* continuous stream.

## How It Works

Ariadne takes a stream of points (vectors) and computes a "Witness" (the Truncated Signature). This is a fixed-size array of floating-point numbers.

* **Level 1 Terms:** The net displacement (Vector).
* **Level 2 Terms:** The signed area enclosed by the path (Matrix/Tensor). This captures "lead-lag" relationships (e.g., did Pressure rise *before* Temperature, or after?).
* **Level 3+ Terms:** Higher-order volumes.

This "Witness" array can then be fed directly into:

* **Neural Networks:** As a hyper-compressed input feature vector that beats raw RNNs on time-series data.
* **Control Systems:** To detect "roughness" or oscillation regimes that simple PID controllers miss.
* **Anomaly Detection:** A sudden change in the "Area" term indicates a regime change in the system's behavior, even if the mean and variance look normal.

## Usage (Zig)

Ariadne is designed for finite problems: No hidden heap allocations, computable memory footprints, and strict typing.

```zig
const std = @import("std");
const ariadne = @import("ariadne");

pub fn main() void {
    // 1. Define a Witness for a 3-dimensional signal (x, y, z)
    //    looking at "Roughness" depth 2 (keeps track of Displacement + Area).
    const Dim = 3;
    const Depth = 2;
    const Witness = ariadne.Signature(Dim, Depth);

    var path_sig = Witness.identity();

    // 2. Feed it a stream of noisy data
    // (Imagine these are readings from a 3-axis accelerometer)
    const points = [_][3]f64{
        .{0.0, 0.0, 0.0},
        .{1.5, 0.2, -0.1},
        .{1.6, 0.9, 0.2},
        .{0.1, 0.0, 0.0}, // A loop!
    };

    var prev = points[0];
    for (points[1..]) |curr| {
        // Calculate the increment
        const delta = Witness.diff(prev, curr);

        // Update the path signature
        path_sig = path_sig.join(delta);

        prev = curr;
    }

    // 3. Inspect the Geometry
    // The "Area" terms will be non-zero, proving a loop occurred.
    std.debug.print("Path Geometry: {any}\n", .{path_sig.coeffs});
}
```

## Performance

* **Zero Allocation:** The `Signature` struct is a fixed-size array derived at `comptime`.
* **SIMD Friendly:** The tensor operations are dense linear algebra, ideal for compiler vectorization.
* **No Dependencies:** Pure Zig.

## Status

**Experimental.** This is a from-scratch implementation of the tensor algebra required for Rough Path Theory, built for engineers who want "the code, not the theorem."

## References

* **Primer (Start Here)**
    * Chevyrev, I. & Kormilitzin, A. (2016). *A Primer on the Signature Method in Machine Learning.*
    * [arXiv:1603.03788](https://arxiv.org/abs/1603.03788/)
* **Theory (Rigor)**
    * Lyons, T. (1998). *Differential equations driven by rough signals.*
    * [DOI: 10.4171/RMI/240](https://doi.org/10.4171/RMI/240/)
* **Modern Approaches**
    * Kidger, P. & Lyons, T. (2020). *Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU.*
    * [arXiv:2001.00706](https://arxiv.org/abs/2001.00706/)
