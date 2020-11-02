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
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.sequences_old != -1
    }

    /// Set a new count value for TimeStamp. This also changes
    /// `sequences_old` to 0 to indicate the TimeStamp belongs to the
    /// current processing sequence.
    ///
    /// # Arguments
    ///
    /// * `newval` - New count value.
    #[inline]
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
/// demodulation signal. This is applied after the frequency scaling
/// factor multiplies the reference frequency.
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
#[inline]
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
#[inline]
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
#[inline]
fn iq_to_a(i: f32, q: f32) -> f32 {
    2. * sqrt(pow2(i) + pow2(q))
}

/// Returns angle from in-phase and quadrature signals.
///
/// # Arguments
///
/// `i` - In-phase signal.
/// `q` - Quadrature signal.
#[inline]
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
/// # Generics
///
/// * `N` - Number of ADC samples.
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
    let tref_count: u16 = tstamps_diff(tstamps, overflow_count);
    let mut thetas: [f32; N] = [0.; N];
    let mut theta_count: u16;

    if tstamps[0].sequences_old == 0 {
        theta_count = (tref_count - first_t) % tref_count;
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
        ) % tref_count;
    }

    let tdemod_count: f32 = tref_count as f32 / fscale as f32;
    let phi_count: f32 = phi / (2. * PI) * tdemod_count;
    thetas[0] = real_phase(theta_count, tdemod_count, phi);
    for i in 1..N {
        theta_count += tadc;
        thetas[i] = real_phase(theta_count, tdemod_count, phi_count);
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
/// * `phase_count` - Phase offset. In the same units as `period_count`.
fn real_phase(theta_count: u16, period_count: f32, phase_count: f32) -> f32 {
    let total_angle = (theta_count as f32 + phase_count) % period_count;
    2. * PI * (total_angle / period_count)
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
    let mut i: [f32; N] = [0.; N];
    let mut q: [f32; N] = [0.; N];

    for n in 0..N {
        let xf_n: f32 = x[n] as f32;
        i[n] = xf_n * sines[n];
        q[n] = xf_n * cosines[n];
    }
    (i, q)
}

#[inline]
fn sqrt(x: f32) -> f32 {
    unsafe { intrinsics::sqrtf32(x) }
}

#[inline]
fn pow2(x: f32) -> f32 {
    x * x
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn abs(x: f32) -> f32 {
        if x >= 0. {
            x
        } else {
            -x
        }
    }

    fn max(x: f32, y: f32) -> f32 {
        if x > y {
            x
        } else {
            y
        }
    }

    fn f32_is_close(a: f32, b: f32) -> bool {
        abs(a - b) <= (max(a, b) * f32::EPSILON)
    }

    #[test]
    fn arr_n_16_ffast_1e6_fadc_5e5() {
        let ffast: u32 = 100_000_000;
        let fadc: u32 = 500_000;
        let n: u16 = 16;
        assert_eq!(arr(ffast, fadc, n), 3200);
    }

    #[test]
    fn decimate_n8_k1() {
        let i_in: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let q_in: [f32; 8] = [0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
        assert!(decimate::<8, 1>(i_in, q_in) == ([0.1], [0.9]));
    }

    #[test]
    fn decimate_n8_k2() {
        let i_in: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let q_in: [f32; 8] = [0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
        assert!(decimate::<8, 2>(i_in, q_in) == ([0.1, 0.5], [0.9, 0.13]));
    }

    #[test]
    fn decimate_n8_k4() {
        let i_in: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let q_in: [f32; 8] = [0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16];
        assert!(decimate::<8, 4>(i_in, q_in) == ([0.1, 0.3, 0.5, 0.7], [0.9, 0.11, 0.13, 0.15]));
    }

    #[test]
    fn real_phase_per_1000_phi_0() {
        let period_count: f32 = 1000.;
        let phi: f32 = 0.;
        // PI/4 increments
        assert!(f32_is_close(real_phase(0, period_count, phi), 0.));
        assert!(f32_is_close(real_phase(125, period_count, phi), PI / 4.));
        assert!(f32_is_close(real_phase(250, period_count, phi), PI / 2.));
        assert!(f32_is_close(
            real_phase(375, period_count, phi),
            3. * PI / 4.
        ));
        assert!(f32_is_close(real_phase(500, period_count, phi), PI));
        assert!(f32_is_close(
            real_phase(625, period_count, phi),
            5. * PI / 4.
        ));
        assert!(f32_is_close(
            real_phase(750, period_count, phi),
            3. * PI / 2.
        ));
        assert!(f32_is_close(
            real_phase(875, period_count, phi),
            7. * PI / 4.
        ));
        assert!(f32_is_close(real_phase(1000, period_count, phi), 0.));
        // other, < 1000
        assert!(f32_is_close(
            real_phase(1, period_count, phi),
            6.28318530718e-3
        ));
        assert!(f32_is_close(
            real_phase(7, period_count, phi),
            0.0439822971503
        ));
        assert!(f32_is_close(
            real_phase(763, period_count, phi),
            4.79407038938
        ));
        // > 1000
        for angle_count in 0..period_count as usize - 1 {
            for p in 0..3 {
                assert!(f32_is_close(
                    real_phase(
                        (angle_count as f32 + p as f32 * period_count) as u16,
                        period_count,
                        phi
                    ),
                    real_phase(angle_count as u16, period_count, phi)
                ));
            }
        }
    }

    #[test]
    fn real_phase_per_1000_phi_adjust() {
        let period_count: f32 = 1000.;
        for theta in [0, 20, 611, 987].iter() {
            for phi in 0..period_count as usize - 1 {
                assert!(f32_is_close(
                    real_phase(*theta, period_count, phi as f32),
                    real_phase(*theta + phi as u16, period_count, 0.)
                ))
            }
        }
    }

    #[test]
    fn increment_tstamp_sequence_valid_invalid() {
        let mut tstamps = [
            TimeStamp {
                count: 0,
                sequences_old: 0,
            },
            TimeStamp {
                count: 0,
                sequences_old: -1,
            },
        ];
        increment_tstamp_sequence(&mut tstamps);
        assert_eq!(tstamps[0].sequences_old, 1);
        assert_eq!(tstamps[1].sequences_old, -1);
    }

    #[test]
    fn tstamps_valid_count_test() {
        for (valid_num, old1, old2) in [(0, -1, -1), (1, 1, -1), (1, -1, 1), (2, 1, 1)].iter() {
            assert_eq!(
                tstamps_valid_count(&[
                    TimeStamp {
                        count: 5,
                        sequences_old: *old1 as i16,
                    },
                    TimeStamp {
                        count: 0,
                        sequences_old: *old2 as i16,
                    }
                ]),
                *valid_num as usize
            );
        }
    }

    #[test]
    fn tadc_test() {
        assert_eq!(tadc(100_000_000, 500_000), 200);
        assert_eq!(tadc(125_000_000, 500_000), 250);
        assert_eq!(tadc(100_000_000, 300_000), 333);
    }

    #[test]
    fn iq_to_a_test() {
        assert!(f32_is_close(iq_to_a(1. / 2f32.sqrt(), 1. / 2f32.sqrt()), 2.));
        assert!(f32_is_close(iq_to_a(0.1, 1.6), 3.20624390838));
        assert!(f32_is_close(iq_to_a(-0.1, 1.6), 3.20624390838));
        assert!(f32_is_close(iq_to_a(0.1, -1.6), 3.20624390838));
    }
}
