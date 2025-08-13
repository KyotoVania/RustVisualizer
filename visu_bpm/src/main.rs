use anyhow::{Context, Result};
use clap::Parser;
use hound::{SampleFormat, WavReader};
use num_complex::Complex;
use rubato::{FftFixedIn, Resampler};
use rustfft::FftPlanner;
use std::path::PathBuf;

/// Programme de détection de BPM utilisant l'autocorrélation et le flux spectral médian
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Chemin vers le fichier WAV à analyser
    #[arg()]
    file_path: PathBuf,

    /// BPM minimum à rechercher
    #[arg(long, default_value_t = 60.0)]
    min_bpm: f32,

    /// BPM maximum à rechercher
    #[arg(long, default_value_t = 180.0)]
    max_bpm: f32,
}

// ====================
// Module Decoder
// ====================
mod decoder {
    use super::*;

    /// Charge et décode un fichier WAV en mono avec rééchantillonnage à 44100 Hz
    pub fn decode_to_mono(path: &PathBuf, target_sr: u32) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path)
            .with_context(|| format!("Impossible d'ouvrir le fichier {:?}", path))?;

        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_sr = spec.sample_rate;

        // Lecture des échantillons
        let samples: Vec<f32> = match spec.sample_format {
            SampleFormat::Float => {
                reader.samples::<f32>()
                    .collect::<Result<Vec<_>, _>>()
                    .context("Erreur lors de la lecture des échantillons")?
            }
            SampleFormat::Int => {
                let bit_depth = spec.bits_per_sample;
                let max_val = (1 << (bit_depth - 1)) as f32;
                reader.samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / max_val))
                    .collect::<Result<Vec<_>, _>>()
                    .context("Erreur lors de la conversion des échantillons")?
            }
        };

        // Conversion en mono
        let mono_samples: Vec<f32> = if channels > 1 {
            samples.chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };

        // Rééchantillonnage si nécessaire
        if source_sr != target_sr {
            resample(&mono_samples, source_sr, target_sr)
        } else {
            Ok(mono_samples)
        }
    }

    /// Rééchantillonne le signal audio à la fréquence cible
    fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
        let mut resampler = FftFixedIn::<f32>::new(
            source_sr as usize,
            target_sr as usize,
            1024,
            1,
            1,
        ).context("Erreur lors de l'initialisation du resampler")?;

        let mut output = vec![vec![0.0f32; resampler.output_frames_max()]; 1];
        let mut resampled = Vec::new();

        // Traitement par chunks
        for chunk in samples.chunks(1024) {
            let input = vec![chunk.to_vec()];
            let (_, out_len) = resampler.process_into_buffer(&input, &mut output, None)
                .context("Erreur lors du rééchantillonnage")?;
            resampled.extend_from_slice(&output[0][..out_len]);
        }

        Ok(resampled)
    }
}

// ====================
// Module ODF (Onset Detection Function)
// ====================
mod odf {
    use super::*;

    /// Paramètres pour la génération de l'ODF
    pub struct OdfGenerator {
        pub frame_size: usize,
        pub hop_size: usize,
    }

    impl OdfGenerator {
        pub fn new(frame_size: usize, hop_size: usize) -> Self {
            Self { frame_size, hop_size }
        }

        /// Génère l'ODF en utilisant le flux spectral agrégé par la médiane
        pub fn generate(&self, samples: &[f32]) -> Vec<f32> {
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(self.frame_size);

            // Création de la fenêtre de Hann manuellement
            let window: Vec<f32> = (0..self.frame_size)
                .map(|i| {
                    let t = i as f32 / (self.frame_size - 1) as f32;
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())
                })
                .collect();

            let mut odf = Vec::new();
            let mut prev_magnitudes = vec![0.0; self.frame_size / 2 + 1];

            // Calcul du nombre de frames
            let num_frames = (samples.len() - self.frame_size) / self.hop_size + 1;

            for frame_idx in 0..num_frames {
                let start = frame_idx * self.hop_size;
                let end = start + self.frame_size;

                if end > samples.len() {
                    break;
                }

                // Application de la fenêtre et préparation pour FFT
                let mut buffer: Vec<Complex<f32>> = samples[start..end]
                    .iter()
                    .zip(window.iter())
                    .map(|(s, w)| Complex::new(s * w, 0.0))
                    .collect();

                // FFT
                fft.process(&mut buffer);

                // Calcul des magnitudes (seulement la moitié positive du spectre)
                let magnitudes: Vec<f32> = buffer[..self.frame_size / 2 + 1]
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                    .collect();

                // Calcul du flux spectral avec agrégation par médiane
                let mut spectral_diffs: Vec<f32> = magnitudes
                    .iter()
                    .zip(prev_magnitudes.iter())
                    .map(|(current, prev)| {
                        // Passage en échelle logarithmique pour plus de robustesse
                        let log_current = (current + 1e-10_f32).ln();
                        let log_prev = (prev + 1e-10_f32).ln();
                        (log_current - log_prev).max(0.0)  // Redressement demi-onde
                    })
                    .collect();

                // Calcul de la médiane
                if !spectral_diffs.is_empty() {
                    spectral_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median_idx = spectral_diffs.len() / 2;
                    odf.push(spectral_diffs[median_idx]);
                } else {
                    odf.push(0.0);
                }

                prev_magnitudes = magnitudes;
            }

            odf
        }
    }
}

// ====================
// Module ACF (Autocorrelation Function)
// ====================
mod acf {
    use super::*;

    /// Analyseur d'autocorrélation utilisant le théorème de Wiener-Khinchin
    pub struct AcfAnalyzer {
        pub window_size: usize,
    }

    impl AcfAnalyzer {
        pub fn new(window_size: usize) -> Self {
            Self { window_size }
        }

        /// Calcule l'autocorrélation à court terme via FFT (Wiener-Khinchin)
        pub fn analyze(&self, odf_buffer: &[f32]) -> Vec<f32> {
            // Prendre les dernières valeurs selon la taille de fenêtre
            let start = odf_buffer.len().saturating_sub(self.window_size);
            let window = &odf_buffer[start..];

            // Zero-padding pour obtenir une corrélation linéaire correcte
            let fft_len = (window.len() * 2).next_power_of_two();

            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(fft_len);
            let ifft = planner.plan_fft_inverse(fft_len);

            // Préparation du buffer avec zero-padding
            let mut buffer: Vec<Complex<f32>> = window
                .iter()
                .map(|&s| Complex::new(s, 0.0))
                .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
                .take(fft_len)
                .collect();

            // 1. Forward FFT
            fft.process(&mut buffer);

            // 2. Power spectrum (|X(f)|²)
            for c in buffer.iter_mut() {
                let power = c.re * c.re + c.im * c.im;
                *c = Complex::new(power, 0.0);
            }

            // 3. Inverse FFT
            ifft.process(&mut buffer);

            // 4. Normalisation et extraction de la partie réelle
            let scale = 1.0 / fft_len as f32;
            buffer.iter()
                .map(|c| c.re * scale)
                .take(window.len())  // Ne garder que la partie utile
                .collect()
        }
    }
}

// ====================
// Module Estimator
// ====================
mod estimator {
    /// Estimateur de BPM avec stabilisation
    pub struct BpmEstimator {
        pub min_bpm: f32,
        pub max_bpm: f32,
        pub history: Vec<f32>,
        pub history_size: usize,
    }

    impl BpmEstimator {
        pub fn new(min_bpm: f32, max_bpm: f32) -> Self {
            Self {
                min_bpm,
                max_bpm,
                history: Vec::new(),
                history_size: 10,
            }
        }

        /// Estime le BPM à partir du vecteur d'autocorrélation
        pub fn estimate(&mut self, acf: &[f32], odf_sample_rate: f32) -> Option<f32> {
            // Conversion BPM en plage de lags
            let max_period_s = 60.0 / self.min_bpm;
            let min_period_s = 60.0 / self.max_bpm;
            let min_lag = (min_period_s * odf_sample_rate).round() as usize;
            let max_lag = (max_period_s * odf_sample_rate).round() as usize;

            if max_lag >= acf.len() || min_lag >= max_lag {
                return None;
            }

            // Recherche du pic maximum dans la plage de lags
            let search_range = &acf[min_lag..=max_lag.min(acf.len() - 1)];
            let (peak_local_idx, &peak_value) = search_range
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

            // Vérification de la qualité du pic (seuil minimal)
            if peak_value < 0.1 {
                return None;
            }

            let peak_lag = min_lag + peak_local_idx;

            // Conversion lag -> BPM
            let bpm = (odf_sample_rate * 60.0) / peak_lag as f32;

            // Stabilisation avec médiane mobile
            self.history.push(bpm);
            if self.history.len() > self.history_size {
                self.history.remove(0);
            }

            // Calcul de la médiane
            let mut sorted_history = self.history.clone();
            sorted_history.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_bpm = sorted_history[sorted_history.len() / 2];

            Some(median_bpm)
        }
    }
}

// ====================
// Programme Principal
// ====================
fn main() -> Result<()> {
    let args = Args::parse();

    println!("Chargement du fichier audio: {:?}", args.file_path);

    // 1. Décodage et prétraitement
    const TARGET_SR: u32 = 44100;
    let samples = decoder::decode_to_mono(&args.file_path, TARGET_SR)?;
    println!("Audio chargé: {} échantillons à {} Hz", samples.len(), TARGET_SR);

    // 2. Génération de l'ODF
    const FRAME_SIZE: usize = 2048;
    const HOP_SIZE: usize = 512;

    let odf_gen = odf::OdfGenerator::new(FRAME_SIZE, HOP_SIZE);
    let odf_signal = odf_gen.generate(&samples);

    let odf_sample_rate = TARGET_SR as f32 / HOP_SIZE as f32;
    println!("ODF générée: {} valeurs à {:.2} Hz", odf_signal.len(), odf_sample_rate);

    // 3. Analyse par autocorrélation
    // Fenêtre de 3 secondes pour l'analyse
    let acf_window_size = (3.0 * odf_sample_rate) as usize;
    let acf_analyzer = acf::AcfAnalyzer::new(acf_window_size);

    // 4. Estimation du BPM
    let mut bpm_estimator = estimator::BpmEstimator::new(args.min_bpm, args.max_bpm);

    // Analyse sur plusieurs fenêtres pour plus de robustesse
    let analysis_hop = (odf_sample_rate as usize) / 2; // Analyse toutes les 0.5 secondes
    let mut final_bpm = None;

    for window_start in (0..odf_signal.len()).step_by(analysis_hop) {
        let window_end = (window_start + acf_window_size).min(odf_signal.len());
        if window_end - window_start < acf_window_size / 2 {
            break; // Fenêtre trop petite
        }

        let acf = acf_analyzer.analyze(&odf_signal[..window_end]);
        if let Some(bpm) = bpm_estimator.estimate(&acf, odf_sample_rate) {
            final_bpm = Some(bpm);
        }
    }

    // Affichage du résultat
    match final_bpm {
        Some(bpm) => println!("BPM: {:.1}", bpm),
        None => eprintln!("Impossible de détecter le BPM"),
    }

    Ok(())
}