#![feature(min_const_generics)]

extern crate stabilizer_lockin;
extern crate std;

// // TODO remove
// use std::fs::File;
// use std::io::prelude::*;
// use std::path::Path;

use std::f64::consts::PI;
use std::vec::Vec;

use stabilizer_lockin::iir::{IIRState, IIR};
use stabilizer_lockin::{postfilt_at, TimeStamp};

const ADC_MAX: f64 = 1.;
const ADC_MAX_COUNTS: f64 = i16::MAX as f64;

// 16-bit ADC has a min dBFS for each sample of -90.
struct PureSine {
    freq: f64,
    amp_dbfs: f64,
    phi: f64,
}

fn dbfs_to_linear(dbfs: f64) -> f64 {
    let base = 10.0_f64;
    ADC_MAX * base.powf(dbfs / 20.)
}

fn linear_to_dbfs(linear: f64) -> f64 {
    20. * (linear / ADC_MAX).log10()
}

/// Frequency (in Hz) to period in terms of the fast clock period.
fn freq_to_tcounts(freq: f64, ffast: f64) -> f64 {
    ffast / freq
}

fn adc_counts(fadc: f64, ffast: f64) -> f64 {
    ffast / fadc
}

fn real_to_adc_sample(x: f64) -> i16 {
    let max: i64 = i16::MAX as i64;
    let min: i64 = i16::MIN as i64;

    // clip inputs
    if (x as i64) > max {
        return i16::MAX;
    } else if (x as i64) < min {
        return i16::MIN;
    }

    (i16::MAX as f64 * x) as i16
}

/// Generate `N` values of an input signal starting at `tstart`.
///
/// # Arguments
///
/// * `pure_sigs` - Sinusoidal components of input signal.
/// * `tstart` - Starting time of input signal in terms of fast clock
/// counts.
fn input_signal<const N: usize>(
    pure_sigs: &Vec<PureSine>,
    tstart: u64,
    ffast: f64,
    fadc: f64,
) -> [i16; N] {
    let mut sig: [i16; N] = [0; N];
    let mut amplitudes: Vec<f64> = Vec::<f64>::new();
    let mut theta_starts: Vec<f64> = Vec::<f64>::new();
    let mut theta_incs: Vec<f64> = Vec::<f64>::new();

    for elem in pure_sigs.iter() {
        let elem_period = freq_to_tcounts(elem.freq, ffast);
        let phi_counts = elem.phi / (2. * PI) * elem_period;
        let theta_start_counts = (phi_counts + tstart as f64) % elem_period;

        amplitudes.push(dbfs_to_linear(elem.amp_dbfs));
        theta_starts.push(2. * PI * theta_start_counts / elem_period);
        theta_incs.push(2. * PI * adc_counts(fadc, ffast) / elem_period);
    }

    for n in 0..N {
        let mut sigf_n: f64 = 0.;
        for i in 0..pure_sigs.len() {
            sigf_n += amplitudes[i] * (theta_starts[i] + theta_incs[i] * n as f64).sin();
        }
        sig[n] = real_to_adc_sample(sigf_n);
    }

    sig
}

/// Reference clock timestamp values in `N` ADC periods starting at
/// `tstart`. Also returns the number of valid timestamps.
fn tstamps<const M: usize>(
    fref: f64,
    phi: f64,
    tstart: u64,
    tstop: u64,
    ffast: f64,
) -> (usize, [u16; M]) {
    // counts in one reference period
    let tref = ffast / fref;
    let phi_counts = (phi / (2. * PI)) * tref;
    let start_counts = (tstart as f64 + phi_counts) % tref;
    let mut tval = (tref - start_counts) % tref;
    let tdist: f64 = (tstop - tstart) as f64;
    let mut r: usize = 0;
    let mut t: [u16; M] = [0; M];

    while tval < tdist {
        t[r] = tval as u16;
        tval += tref;
        r += 1;
    }

    (r, t)
}

/// Lowpass biquad filter using cutoff and sampling frequencies.
/// Taken from: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
fn lp_iir_ba(fc: f64, fs: f64) -> [f32; 5] {
    let w0: f64 = 2. * PI * fc / fs;
    let q: f64 = 1. / 2f64.sqrt();
    let alpha: f64 = w0.sin() / (2. * q);
    let mut b0: f64 = (1. - w0.cos()) / 2.;
    let mut b1: f64 = 1. - w0.cos();
    let mut b2: f64 = b0;
    let a0: f64 = 1. + alpha;
    let mut a1: f64 = -2. * w0.cos();
    let mut a2: f64 = 1. - alpha;
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= -a0;
    a2 /= -a0;

    [b0 as f32, b1 as f32, b2 as f32, a1 as f32, a2 as f32]
}

/// Check that a measured value is within some tolerance of the actual
/// value. Setting `reference` to the full scale value gives a full
/// scale uncertainty. Setting `reference` to `act` uses a percentage
/// accuracy.
fn tol_check(act: f32, res: f32, fixed_tol: f32, rel_tol: f32) -> bool {
    (act - res).abs() < max_error(act, fixed_tol, rel_tol)
}

fn max_error(act: f32, fixed_tol: f32, rel_tol: f32) -> f32 {
    rel_tol * act.abs() + fixed_tol
}

fn lp_test<const N: usize, const M: usize, const K: usize>(
    ffast: f64,
    fadc: f64,
    fref: f64,
    fscale: u16,
    fc: f64,
    desired_input: PureSine,
    noise_inputs: &mut Vec<PureSine>,
    tau_factor: f64,
    tol: f32,
) {
    let sample_counts: u64 = (ffast / fadc) as u64 * N as u64;

    let tau: f64 = 1. / (2. * PI * fc);
    let n_samples = (tau_factor * tau * fadc) as usize;
    // Ensure stability after `tau_factor` time constants.
    let extra_samples = (tau * fadc) as usize;

    let in_dbfs: f64 = desired_input.amp_dbfs;
    let in_a: f64 = dbfs_to_linear(in_dbfs);
    let in_phi: f64 = desired_input.phi;
    let mut in_a_noise: f64 = 0.;
    let mut in_phi_noise: f64 = 0.;

    for noise_input in noise_inputs.iter() {
        // Noise inputs create an oscillation at the output, where the
        // oscillation magnitude is determined by the strength of the
        // noise and its attenuation (attenuation is determined by its
        // proximity to the demodulation frequency and filter rolloff)
        let octaves = ((noise_input.freq - (fref * fscale as f64)).abs() / fc).log2();
        let attenuation = -12. * octaves;
        let noise_lin = dbfs_to_linear(noise_input.amp_dbfs + attenuation);
        in_a_noise += noise_lin;
        // Noise affects the phase output by creating oscillations to
        // I and Q, which affects atan2(Q, I).
        // TODO I'm not so sure about this...
        let phi_err = {
            let i = in_a / 2. * in_phi.cos();
            let q = in_a / 2. * in_phi.sin();
            ((q + noise_lin).atan2(i - noise_lin) - q.atan2(i)).abs()
        };
        in_phi_noise += phi_err;
    }

    let pure_sigs = noise_inputs;
    pure_sigs.push(desired_input);

    let iir = IIR {
        ba: lp_iir_ba(fc, fadc),
        y_offset: 0.,
        y_min: -f32::INFINITY,
        y_max: f32::INFINITY,
    };

    let iirs: [IIR; 2] = [iir, iir];
    let mut iir_states: [IIRState; 2] = [[0.; 5], [0.; 5]];

    // let path = Path::new("log.txt");
    // let display = path.display();

    // let mut file = match File::create(&path) {
    //     Err(why) => panic!("failed to write to {}: {}", display, why),
    //     Ok(file) => file,
    // };

    let mut timestamps = [
        TimeStamp { count: 0, sequences_old: -1 },
        TimeStamp { count: 0, sequences_old: -1 }
    ];

    let in_a: f32 = in_a as f32;
    let in_phi: f32 = in_phi as f32;

    for n in 0..(n_samples + extra_samples) {
        let tstart: u64 = n as u64 * sample_counts;
        let sig: [i16; N] = input_signal::<N>(&pure_sigs, tstart, ffast, fadc);
        let (r, ts) = tstamps::<M>(fref, 0., tstart, tstart + sample_counts - 1, ffast);
        let (a, t) = postfilt_at::<N, M, K>(
            sig,
            ts,
            r,
            0.,
            ffast as u32,
            fadc as u32,
            fscale,
            iirs,
            &mut iir_states,
            &mut timestamps,
        );

        // if n == n_samples {
        //     match file.write_all("\npost tau_factor\n\n".as_bytes()) {
        //         Err(why) => panic!("failed to write to {}: {}", display, why),
        //         Ok(_) => (),
        //     }
        // }

        // for k in 0..K as usize {
        //     let s_str = format!("{:.6}\t{:.6}\n", linear_to_dbfs(a[k] as f64 / ADC_MAX_COUNTS as f64), t[k]);
        //     match file.write_all(s_str.as_bytes()) {
        //         Err(why) => panic!("failed to write to {}: {}", display, why),
        //         Ok(_) => (),
        //     }
        // }

        // Ensure stable below tolerance for 1 time constant after `tau_factor`.
        if n >= n_samples {
            for k in 0..K {
                let a_norm: f32 = a[k] / ADC_MAX_COUNTS as f32;
                assert!(
                    tol_check(in_a, a_norm, in_a_noise as f32, tol),
                    "a_act: {:.4} ({:.2} dBFS), a_meas: {:.4} ({:.2} dBFS), tol: {:.4}",
                    in_a,
                    in_dbfs,
                    a_norm,
                    linear_to_dbfs(a_norm as f64),
                    max_error(in_a, in_a_noise as f32, tol)
                );
                assert!(
                    tol_check(in_phi, t[k], in_phi_noise as f32, tol),
                    "t_act: {:.4}, t_meas: {:.4}, tol: {:.4}",
                    in_phi,
                    t[k],
                    max_error(in_phi, in_phi_noise as f32, tol)
                );
            }
        }
    }
}

#[test]
fn lp_fundamental_sideband_noise_phi_0() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 100e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<16, 8, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.1 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.9 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_fundamental_sideband_noise_phi_pi_2() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 100e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<16, 8, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: PI / 2.,
        },
        &mut vec![
            PureSine {
                freq: 1.1 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.9 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_fundamental_sideband_noise_k_4() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 100e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<16, 8, 4>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.1 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.9 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_fundamental_sideband_noise_no_downsample() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 100e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<16, 8, 16>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.1 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.9 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_fundamental_111e3_sideband_noise_phi_pi_4() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 111e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<16, 8, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: PI / 4.,
        },
        &mut vec![
            PureSine {
                freq: 1.1 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.9 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_first_harmonic_sideband_noise() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 50e3;
    let fscale: u16 = 2;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.2 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.8 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_second_harmonic_sideband_noise() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 50e3;
    let fscale: u16 = 3;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.2 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.8 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_third_harmonic_sideband_noise() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 50e3;
    let fscale: u16 = 4;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.2 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.8 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_first_harmonic_phase_shift() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 50e3;
    let fscale: u16 = 2;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: PI / 4.,
        },
        &mut vec![
            PureSine {
                freq: 1.2 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.8 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_fadc_1e6() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 1e6;
    let fsig: f64 = 100e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.2 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.8 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_ffast_125e6() {
    let ffast: f64 = 125e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 100e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 1.2 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
            PureSine {
                freq: 0.8 * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            },
        ],
        tau,
        tol,
    )
}

#[test]
fn lp_low_t() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 10e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut vec![
            PureSine {
                freq: 2. * fdemod,
                amp_dbfs: -20.,
                phi: 0.,
            }
        ],
        tau,
        tol,
    )
}

// TODO this fails because ffast / fsig > u16::MAX. Can fix by
// replacing u16 with u32.
#[test]
fn lp_very_low_t() {
    let ffast: f64 = 100e6;
    let fadc: f64 = 500e3;
    let fsig: f64 = 1e3;
    let fscale: u16 = 1;
    let fc: f64 = 1e3;
    let fdemod: f64 = fscale as f64 * fsig;
    let tau: f64 = 5.;
    let tol: f32 = 1e-2;

    lp_test::<32, 16, 1>(
        ffast,
        fadc,
        fsig,
        fscale,
        fc,
        PureSine {
            freq: fdemod,
            amp_dbfs: -30.,
            phi: 0.,
        },
        &mut Vec::<PureSine>::new(),
        tau,
        tol,
    )
}
