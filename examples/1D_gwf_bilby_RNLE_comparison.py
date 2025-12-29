#!/home/mattia.emma/.conda/envs/sbi/bin/python

import sys
import time
from collections import namedtuple
import json
import argparse
import os
import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import copy
import pickle


import sbilby
from sbilby.simulation_based_inference_multidetector import GenerateData
from sbilby.simulation_based_inference_multidetector import AdditiveSignalAndNoise
import matplotlib.pyplot as plt
from sbilby.data_generation import GenerateWhitenedIFONoise_fromGWF
from sbilby.data_generation import GenerateWhitenedSignal_fromGWF

class BenchmarkLikelihood(object):
    def __init__(
        self,
        benchmark_likelihood,
        reference_likelihood,
        prior,
        reference_prior,
        outdir,
        injection_parameters,
        training_time,
    ):
        self.benchmark_likelihood = benchmark_likelihood
        self.reference_likelihood = reference_likelihood
        self.prior = prior
        self.reference_prior = reference_prior
        self.outdir = outdir
        self.injection_parameters = injection_parameters
        self.statistics = dict(
            likelihood_class=benchmark_likelihood.__class__.__name__,
            likelihood_metadata=benchmark_likelihood.meta_data,
        )
        self.training_time=training_time

    def _time_likelihood(self, likelihood, n, name):
        eval_times = []
        for _ in range(n):
            likelihood.parameters.update(self.prior.sample())
            start = time.time()
            likelihood.log_likelihood()
            end = time.time()
            eval_times.append(end - start)
        self.statistics[f"likelihood_{name}_eval_time_mean"] = float(
            np.mean(eval_times)
        )
        self.statistics[f"likelihood_{name}_eval_time_std"] = float(np.std(eval_times))

    def benchmark_time(self, n=100):
        self._time_likelihood(self.benchmark_likelihood, n, "benchmark")
        self._time_likelihood(self.reference_likelihood, n, "reference")
        self.statistics["likelihood_training_time"]=self.training_time
    def benchmark_posterior_sampling(self, run_sampler_kwargs=None):
        kwargs = dict(nlive=1000, sampler="dynesty", dlogz=0.5, check_point_delta_t=60)
        if run_sampler_kwargs is not None:
            kwargs.update(run_sampler_kwargs)

        result_reference = bilby.run_sampler(
            likelihood=self.reference_likelihood,
            priors=self.reference_prior,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.benchmark_likelihood.label + "_REFERENCE",
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            **kwargs,
        )
        
        result_benchmark = bilby.run_sampler(
            likelihood=self.benchmark_likelihood,
            priors=self.prior,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.benchmark_likelihood.label,
            #conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            **kwargs,
        )

        for key, val in self.reference_prior.items():
            if val.is_fixed is False:
                samplesA = result_benchmark.posterior[key]
                samplesB = result_reference.posterior[key]
                js = calculate_js(samplesA, samplesB)
                self.statistics[f"1D_posterior_JS_{key}_median"] = js.median
                self.statistics[f"1D_posterior_JS_{key}_plus"] = js.plus
                self.statistics[f"1D_posterior_JS_{key}_minus"] = js.minus

                fig, ax = plt.subplots()
                ax.hist(samplesA, bins=50, alpha=0.8, label="RNLE", density=True)
                ax.hist(samplesB, bins=50, alpha=0.8, label="Bilby", density=True)
                ax.axvline(self.injection_parameters[key], color="k")
                ax.set(xlabel=key, title=f"JS={js.median}")
                ax.legend()
                plt.savefig(
                        f"{self.outdir}/{self.benchmark_likelihood.label}_1D_posterior_{key}.png"
                )
                    

    def write_results(self):
        bilby.utils.check_directory_exists_and_if_not_mkdir("RESULTS_working_sampling_Greg_whitened")
        with open(
            f"RESULTS_working_sampling_Greg_whitened/result_benchmark_{self.benchmark_likelihood.label}.json", "w"
        ) as file:
            json.dump(self.statistics, file, indent=4)

def calc_summary(jsvalues, quantiles=(0.16, 0.84)):
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(jsvalues, quants_to_compute * 100)
    summary = namedtuple("summary", ["median", "lower", "upper"])
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]
    return summary


def calculate_js(samplesA, samplesB, ntests=100, xsteps=100):
    js_array = np.zeros(ntests)
    for j in range(ntests):
        nsamples = min([len(samplesA), len(samplesB)])
        A = np.random.choice(samplesA, size=nsamples, replace=False)
        B = np.random.choice(samplesB, size=nsamples, replace=False)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)
          
        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf), 2))

    return calc_summary(js_array)






print(f"Running command {' '.join(sys.argv)}")

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", type=int,default=1)
parser.add_argument("--likelihood", type=str, default="RNLE")
parser.add_argument("--num-simulations", type=int, default=20)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--nlive", type=int, default=1500)
parser.add_argument("--dlogz", type=float, default=0.1)
parser.add_argument("--rseed", type=int, default=42)
parser.add_argument("--time_lower", type=float, default=0)
parser.add_argument("--time_upper", type=float, default=0)
parser.add_argument("--fs", type=float, default=4096)
args = parser.parse_args()

# use_mask=False
# outdir = "outdir_benchmark_gw_sampling_frequency/Runs_working_L_"+args.likelihood+'_N'+str(args.num_simulations)+'_D'+str(args.dimensions)+'_TL'+str(args.time_lower)+'_TU'+str(args.time_upper)+'mask_'+str(use_mask)
np.random.seed(args.rseed)
times=[args.time_lower, args.time_upper]
num_simulations = args.num_simulations
################################################# Old code ######################################################
interferometers=["H1","L1"]  
# Standard Bilby
injection_parameters = dict(
    chirp_mass=28.0,
    mass_ratio=1.0,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.000000,
    ra=1.375,
    dec=-1.2108,
)

signal_priors = bilby.gw.prior.BBHPriorDict()
signal_priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=20, maximum=40, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
signal_priors['geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters["geocent_time"]-0.1,maximum=injection_parameters["geocent_time"]+0.1)
for key in [
    "a_1",
    "a_2",
    #"chirp_mass",
    "mass_ratio",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "phase",
    "geocent_time",
    "luminosity_distance",
    "theta_jn"
]:
    signal_priors[key] = injection_parameters[key]
noise_priors = bilby.core.prior.PriorDict(dict(sigma=bilby.core.prior.Uniform(0, 2, 'sigma')))

duration = 4.0
sampling_frequency = args.fs #4096
minimum_frequency = 20
start_time = injection_parameters["geocent_time"] - duration / 2

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

outdir = "outdir_greg_sampling_whitened"


ifo_data = bilby.gw.detector.InterferometerList(["H1"])
ifo_data[0].set_strain_data_from_frame_file('Test_aligned_4096.gwf',channel='H1:1126259642_SIM' ,sampling_frequency=sampling_frequency, duration= duration, start_time=start_time, buffer_time=4)

ifo_signal = bilby.gw.detector.InterferometerList(["H1"])
ifo_signal[0].set_strain_data_from_frame_file('Test_aligned_noise_4096.gwf',channel='H1:1126259642_SIM' ,sampling_frequency=sampling_frequency, duration= duration, start_time=start_time, buffer_time=4)
ifo_noise = bilby.gw.detector.InterferometerList(["H1"])
ifo_noise[0].set_strain_data_from_frame_file('Test_aligned_noise_4096.gwf',channel='H1:1126259642_SIM' ,sampling_frequency=sampling_frequency, duration= duration, start_time=start_time, buffer_time=4)

reference_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifo_data, waveform_generator)

# Extract observational data
yobs = ifo_data[0].whitened_time_domain_strain
xobs = ifo_data[0].time_array

# SBI setup
genA_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)
# genA_ifos = bilby.gw.detector.InterferometerList(["H1"])
# genA_ifos.set_strain_data_from_power_spectral_densities(
#     sampling_frequency=sampling_frequency,
#     duration=duration,
#     start_time=start_time)

genB_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)


priors = noise_priors | signal_priors
priors = bilby.core.prior.PriorDict(copy.deepcopy(priors))
priors.convert_floats_to_delta_functions()

use_mask = True
#times=[0.5, 0.1]
full_noise = GenerateWhitenedIFONoise_fromGWF(ifo_noise[0], use_mask, times)
full_signal = GenerateWhitenedSignal_fromGWF(ifo_signal[0], waveform_generator, copy.deepcopy(signal_priors), use_mask, times)
full_signal_and_noise = AdditiveSignalAndNoise(full_signal, full_noise)

reference_priors = copy.deepcopy(signal_priors)
reference_priors = bilby.core.prior.PriorDict(reference_priors)
reference_priors.convert_floats_to_delta_functions()

#Training the neural network 
label = f"N{num_simulations}_fs{sampling_frequency}_seed{args.rseed}_R{args.repeat}_TL{args.time_lower}_TU{args.time_upper}"
if os.path.exists("../likelihood_cache/"+label+".pkl"):
    os.remove("../likelihood_cache/"+label+".pkl")
else:
    print("File not found!")
benchmark_likelihood = sbilby.simulation_based_inference_multidetector.NLEResidualLikelihood(
        yobs,
        full_signal_and_noise,
        interferometers,
        bilby_prior=copy.deepcopy(priors),
        label=label,
        num_simulations=num_simulations,
        cache_directory='likelihood_cache',
        cache=True,
        #show_progress_bar=True,
)

start=time.time()
benchmark_likelihood.init()
end=time.time()
training_time=end-start

# Need this as the cutting is not applied to the generator yobs yet
self = full_signal
window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
benchmark_likelihood.yobs = benchmark_likelihood.yobs[mask]


######################################   Benchmark ##################################
bench = BenchmarkLikelihood(
    benchmark_likelihood,
    reference_likelihood,
    priors,
    reference_priors,
    outdir,
    injection_parameters=injection_parameters,
    training_time=training_time,
    
)
bench.benchmark_time()
# bench.benchmark_posterior_sampling(
#     dict(
#         sampler="dynesty",
#         nlive=args.nlive,
#         dlogz=args.dlogz,
#         resume=args.resume,
#         print_method="interval-10",
#         npool=8,
#         sample="acceptance-walk",
#         check_point_delta_t=180,
#     )
# )
# bench.write_results()