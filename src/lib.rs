#![no_std]
#![feature(core_intrinsics)]
#![feature(min_const_generics)]

use core::f32::consts::PI;
use core::intrinsics;

pub mod iir;
mod trig;
use trig::{atan2, cos_map, sin_map};

/// Slow external reference edge timestamps.
#[derive(Copy, Clone)]
pub struct TimeStamp {
    // Timestamp value.
    pub count: u16,
    // Number of sequences before the current one that the timestamp
    // occurred. A sequence is a set of `N` ADC samples. E.g., a
    // timestamp from the current sequence has this set to 0, a
    // timestamp from the previous sequence has this set to 1, etc. A
    // value of -1 indicates an invalid timestamp (i.e., one that has
    // not yet been set).
    pub sequences_old: i16,
}

impl TimeStamp {
    /// Increments `sequences_old` if TimeStamp is valid. This is
    /// called at the end of processing a sequence of ADC samples.
    pub fn new_sequence(&mut self) {
        if self.sequences_old != -1 {
            self.sequences_old += 1;
        }
    }

    /// Returns true if TimeStamp is valid.
    pub fn is_valid(&self) -> bool {
        self.sequences_old != -1
    }

    /// Set a new count value for TimeStamp. This also changes
    /// `sequences_old` to 0 to indicate the TimeStamp belongs to the
    /// current processing sequence.
    pub fn new_count(&mut self, newval: u16) {
        self.count = newval;
        self.sequences_old = 0;
    }
}

/// Unfiltered in-phase and quadrature signals.
///
/// # Generics
///
/// * `N` - Number of ADC samples in each processing sequence.
/// This must be a power of 2.
/// * `M` - Maximum number of external reference edge timestamps.
/// The highest this should ever be set is N>>1.
/// * `K` - Number of output samples. This must be a power of 2 and
/// between 1 and `N` (inclusive).
///
/// # Arguments
///
/// * `x` - ADC samples.
/// * `t` - Counter values indicate the timestamps of the slow external
/// reference clock edges.
/// * `r` - Number of valid timestamps in `t`.
/// * `phi` - Demodulation phase offset. This phase shifts the
/// demodulation signal.
/// * `ffast` - Fast clock frequency (Hz). The fast clock increments
/// timestamp counter values used to record the edges of the external
/// reference.
/// * `fadc` - ADC sampling frequency (in Hz).
/// * `fscale` - Scaling factor for the demodulation frequency. For
/// instance, 2 would demodulate with the first harmonic of the reference
/// frequency.
/// * `tstamps_mem` - Last two external reference timestamps (i.e., recorded
/// values of `t`.)
pub fn prefilt<const N: usize, const M: usize, const K: usize>(
    x: [i16; N],
    t: [u16; M],
    r: usize,
    phi: f32,
    ffast: u32,
    fadc: u32,
    fscale: u16,
    tstamps_mem: &mut [TimeStamp; 2],
) -> ([f32; K], [f32; K]) {
    record_new_tstamps::<M>(t, r, tstamps_mem);
    let ts_valid_count = tstamps_valid_count(tstamps_mem);

    if ts_valid_count < 2 {
        return ([0.; K], [0.; K]);
    }

    let tadc = tadc(ffast, fadc);
    let thetas = adc_phases::<N>(t[0], tstamps_mem, phi, fscale, tadc);
    let sines = sin_map::<N>(thetas);
    let cosines = cos_map::<N>(thetas);

    let (i, q) = demod(x, sines, cosines);

    increment_tstamp_sequence(tstamps_mem);

    decimate::<N, K>(i, q)
}

/// Filtered in-phase and quadrature signals.
///
/// # Arguments
///
/// See `prefilt`.
/// * `iir` - IIR biquad for in-phase and quadrature components.
/// * `iirstate` - IIR biquad state for in-phase and quadrature
/// components.
pub fn postfilt_iq<const N: usize, const M: usize, const K: usize>(
    x: [i16; N],
    t: [u16; M],
    r: usize,
    phi: f32,
    ffast: u32,
    fadc: u32,
    fscale: u16,
    iir: [iir::IIR; 2],
    iirstate: &mut [iir::IIRState; 2],
    tstamps_mem: &mut [TimeStamp; 2],
) -> ([f32; K], [f32; K]) {
    record_new_tstamps::<M>(t, r, tstamps_mem);
    let ts_valid_count = tstamps_valid_count(tstamps_mem);

    if ts_valid_count < 2 {
        return ([0.; K], [0.; K]);
    }

    let tadc = tadc(ffast, fadc);
    let thetas = adc_phases::<N>(t[0], tstamps_mem, phi, fscale, tadc);
    let sines = sin_map::<N>(thetas);
    let cosines = cos_map::<N>(thetas);

    let (i, q) = {
        let (id, qd) = demod(x, sines, cosines);
        filter(id, qd, iir, iirstate)
    };

    increment_tstamp_sequence(tstamps_mem);

    decimate::<N, K>(i, q)
}

/// Filtered magnitude and angle signals.
///
/// # Arguments
///
/// See `postfilt_iq`.
pub fn postfilt_at<const N: usize, const M: usize, const K: usize>(
    x: [i16; N],
    t: [u16; M],
    r: usize,
    phi: f32,
    ffast: u32,
    fadc: u32,
    fscale: u16,
    iir: [iir::IIR; 2],
    iirstate: &mut [iir::IIRState; 2],
    tstamps_mem: &mut [TimeStamp; 2],
) -> ([f32; K], [f32; K]) {
    let (i, q) = postfilt_iq::<N, M, K>(
        x,
        t,
        r,
        phi,
        ffast,
        fadc,
        fscale,
        iir,
        iirstate,
        tstamps_mem,
    );

    // TODO we should return here when tstamp valid count < 2. This is
    // unnecessary processing.

    let a = iq_to_a_map::<K>(i, q);
    let t = iq_to_t_map::<K>(i, q);

    (a, t)
}

/// ARR (counter overflow value).
///
/// # Arguments
///
/// * `ffast` - Fast clock frequency (Hz). The fast clock increments
/// timestamp counter values used to record the edges of the external
/// reference.
/// * `fadc` - ADC sampling frequency (in Hz).
/// * `n` - Number of ADC samples in each processing block.
pub fn arr(ffast: u32, fadc: u32, n: u16) -> u16 {
    tadc(ffast, fadc) * n
}

/// Count number of valid TimeStamps from `tstamps`.
fn tstamps_valid_count(tstamps: &[TimeStamp; 2]) -> usize {
    let mut valid_count: usize = 0;
    for i in 0..2 {
        if tstamps[i].is_valid() {
            valid_count += 1;
        }
    }
    valid_count
}

/// Add new timestamps to the TimeStamp memory.
///
/// # Arguments
///
/// * `t` - New timestamp values.
/// * `r` - Number of valid timestamps.
/// * `tstamps_mem` - Last 2 recorded timestamps.
fn record_new_tstamps<const M: usize>(t: [u16; M], r: usize, tstamps_mem: &mut [TimeStamp; 2]) {
    if r > 1 {
        tstamps_mem[1].new_count(t[r - 2]);
        tstamps_mem[0].new_count(t[r - 1]);
    } else if r == 1 {
        tstamps_mem[1].count = tstamps_mem[0].count;
        tstamps_mem[1].sequences_old = tstamps_mem[0].sequences_old;
        tstamps_mem[0].new_count(t[r - 1]);
    }
}

/// ADC period in number of fast clock counts. Assumes `ffast` and
/// `fadc` chosen to yield an integer ratio.
///
/// # Arguments
///
/// * `ffast` - Fast clock frequency.
/// * `fadc` - ADC sampling frequency.
fn tadc(ffast: u32, fadc: u32) -> u16 {
    (ffast / fadc) as u16
}

/// Map `iq_to_a` to each pair of `i` and `q`.
fn iq_to_a_map<const K: usize>(i: [f32; K], q: [f32; K]) -> [f32; K] {
    let mut a: [f32; K] = [0.; K];
    for k in 0..K {
        a[k] = iq_to_a(i[k], q[k]);
    }
    a
}

/// Returns magnitude from in-phase and quadrature signals.
///
/// # Arguments
///
/// `i` - In-phase signal.
/// `q` - Quadrature signal.
fn iq_to_a(i: f32, q: f32) -> f32 {
    2. * sqrt(pow2(i) + pow2(q))
}

/// Returns angle from in-phase and quadrature signals.
///
/// # Arguments
///
/// `i` - In-phase signal.
/// `q` - Quadrature signal.
fn iq_to_t(i: f32, q: f32) -> f32 {
    atan2(q, i)
}

/// Map `iq_to_t` to each pair of `i` and `q`.
fn iq_to_t_map<const K: usize>(i: [f32; K], q: [f32; K]) -> [f32; K] {
    let mut t: [f32; K] = [0.; K];
    for k in 0..K {
        t[k] = iq_to_t(i[k], q[k]);
    }
    t
}

/// Demodulation phase values corresponding to each ADC sample.
///
/// # Arguments
///
/// * `first_t` - First timestamp value from the current processing
/// period. The value provided here doesn't matter if there were no
/// timestamps in the current processing period.
/// * `tstamps` - Recorded TimeStamps.
/// * `phi` - Reference phase offset.
/// * `fscale` - Frequency scaling factor for the demodulation signal.
/// * `tadc` - ADC sampling period.
fn adc_phases<const N: usize>(
    first_t: u16,
    tstamps: &mut [TimeStamp; 2],
    phi: f32,
    fscale: u16,
    tadc: u16,
) -> [f32; N] {
    let overflow_count: u16 = tadc * N as u16;
    let tr_unscaled: u16 = tstamps_diff(tstamps, overflow_count);
    let tr: u16 = tr_unscaled / fscale;
    let mut thetas: [f32; N] = [0.; N];
    let mut theta_count: u16;

    if tstamps[0].sequences_old == 0 {
        theta_count = (tr_unscaled - first_t) % tr;
    } else {
        theta_count = tstamps_diff(
            &[
                TimeStamp {
                    count: 0,
                    sequences_old: 0,
                },
                tstamps[0],
            ],
            overflow_count,
        ) % tr;
    }

    thetas[0] = real_phase(theta_count, tr, phi);
    for i in 1..N {
        theta_count += tadc;
        thetas[i] = real_phase(theta_count, tr, phi);
    }

    thetas
}

/// Number of fast clock counts between two consecutive
/// TimeStamps. This requires that `tstamps[0]` is more recent than
/// `tstamps[1]` but otherwise imposes no restrictions on them. For
/// instance, they can be from different processing periods and this
/// will still count the number of counts between them, accounting for
/// overflow wrapping.
///
/// # Arguments
///
/// * `tstamps` - TimeStamp values.
/// * `overflow_count` - Max timestamp value.
fn tstamps_diff(tstamps: &[TimeStamp; 2], overflow_count: u16) -> u16 {
    if tstamps[0].sequences_old == tstamps[1].sequences_old {
        return tstamps[0].count - tstamps[1].count;
    }

    let rem0: u16 = tstamps[0].count;
    let rem1: u16 = overflow_count - tstamps[1].count;
    let empty_sequences = tstamps[1].sequences_old - tstamps[0].sequences_old - 1;

    rem0 + rem1 + overflow_count * empty_sequences as u16
}

/// Increment `sequences_old` in each TimeStamp of `tstamps`.
fn increment_tstamp_sequence(tstamps: &mut [TimeStamp; 2]) {
    tstamps[0].new_sequence();
    tstamps[1].new_sequence();
}

/// Compute the phase (in radians) for a integral phase given in
/// counts relative to some period in counts.
///
/// # Arguments
///
/// * `theta_count` - Phase in counts. This can be greater than the
/// period in counts.
/// * `period_count` - Number of counts in 1 period.
/// * `phase_offset` - Phase offset (in radians) to add to the real
/// phase result.
fn real_phase(theta_count: u16, period_count: u16, phase_offset: f32) -> f32 {
    (2. * PI * theta_count as f32 / period_count as f32 + phase_offset) % (2. * PI)
}

/// Filter in-phase and quadrature signals with the IIR biquad filter.
///
/// TODO this current does not offer enough filtering flexibility. For
/// instance, we might want to filter with two consecutive biquads.
///
/// # Arguments
///
/// `i` - In-phase signals.
/// `q` - Quadrature signals.
/// `iir` - IIR filters for the in phase (element 0) and quadrature
/// (element 1) signals.
/// `iirstate` - State of each IIR filter.
fn filter<const N: usize>(
    i: [f32; N],
    q: [f32; N],
    iir: [iir::IIR; 2],
    iirstate: &mut [iir::IIRState; 2],
) -> ([f32; N], [f32; N]) {
    let mut filt_i: [f32; N] = [0.; N];
    let mut filt_q: [f32; N] = [0.; N];

    for n in 0..N {
        filt_i[n] = iir[0].update(&mut iirstate[0], i[n]);
        filt_q[n] = iir[1].update(&mut iirstate[1], q[n]);
    }

    (filt_i, filt_q)
}

/// Decimate (downsample) from `N` to `K` samples. N/K is assumed to
/// be equal to 2**n, where n is some non-negative integer. Decimates
/// the in-phase and quadrature signals separately and returns the
/// result as (i, q).
///
/// # Arguments
///
/// `i` - In-phase signal.
/// `q` - Quadrature signal.
fn decimate<const N: usize, const K: usize>(i: [f32; N], q: [f32; N]) -> ([f32; K], [f32; K]) {
    let n_sub_k: usize = N / K;
    let mut res_i: [f32; K] = [0.; K];
    let mut res_q: [f32; K] = [0.; K];
    let mut j: usize = 0;
    let mut k: usize = 0;

    for n in 0..N {
        if j == 0 {
            res_i[k] = i[n];
            res_q[k] = q[n];
            k += 1;
            // Handle no decimation case. TODO there's probably a more
            // efficient way to do this.
            if n_sub_k > 1 {
                j += 1;
            }
        } else {
            if j == n_sub_k - 1 {
                j = 0;
            } else {
                j += 1;
            }
        }
    }

    (res_i, res_q)
}

/// Demodulate ADC inputs with in-phase and quadrature demodulation
/// signals.
///
/// # Arguments
///
/// * `x` - ADC samples.
/// * `sines` - Reference sine signal.
/// * `cosines` - Reference cosine signal.
fn demod<const N: usize>(x: [i16; N], sines: [f32; N], cosines: [f32; N]) -> ([f32; N], [f32; N]) {
    let mut xf: [f32; N] = [0.; N];
    let mut i: [f32; N] = [0.; N];
    let mut q: [f32; N] = [0.; N];

    for n in 0..N {
        xf[n] = x[n] as f32;
        i[n] = xf[n] * sines[n];
        q[n] = xf[n] * cosines[n];
    }
    (i, q)
}

fn sqrt(x: f32) -> f32 {
    unsafe { intrinsics::sqrtf32(x) }
}

fn pow2(x: f32) -> f32 {
    x * x
}

#[cfg(test)]
extern crate std;
mod tests {
    use super::*;

    #[test]
    fn arr_n_16_ffast_1e6_fadc_5e5() {
        let ffast: u32 = 100_000_000;
        let fadc: u32 = 500_000;
        let n: u16 = 16;
        assert_eq!(arr(ffast, fadc, n), 3200);
    }
}