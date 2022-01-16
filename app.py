from flask import Flask, request
import json as json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
app = Flask(__name__)

@app.route('/query-example')
def query_example():
    return 'Query String Example'

@app.route('/form-example')
def form_example():
    return 'Form Data Example'

#Heart of the program

@app.route('/json-example', methods=['POST'])
def json_example():
    request_data = request.get_json()
    print(len(request_data['data']))
    print(request_data)

    out_file = open(f'file{time.time()}.txt', "w")
    json.dump(request_data, out_file)


#Getting data from accelerometer

    f11=request_data
    data11=f11
    ctr=0
    ind_ctr=0
    all_vals=pd.DataFrame([])
    All_data=[data11]
    for data in All_data:
        for i in data['data'][1:]:
            ctr=ctr+1
            ind_ctr=ind_ctr+1
            all_vals=all_vals.append(pd.DataFrame([[i['x'],i['y'],i['z'],i['unix']]],columns=['x','y','z','timestamp']),ignore_index=True)
        print("Readings per second = ",ind_ctr/10)
        ind_ctr=0
    if(len(all_vals)>400):

        all_vals['TS']=np.linspace(0.00015,1,len(all_vals))

        mod_all_vals=all_vals.copy()
        mod_all_vals.drop(columns=['timestamp'],inplace=True)

        A=mod_all_vals.drop(columns=['TS'])

#Removing noise and converting the data to single dimension
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=1)
        S_ = ica.fit_transform(A)

        X=S_.transpose()
        """
        Cepstrum
        ========
    
        """


        __all__ = ['complex_cepstrum', 'real_cepstrum', 'inverse_complex_cepstrum', 'minimum_phase']

#Function used to convert the reduced data to cepstrum values
        def complex_cepstrum(x, n=None):
            r"""Compute the complex cepstrum of a real sequence.

            Parameters
            ----------
            x : ndarray
            Real sequence to compute complex cepstrum of.
            n : {None, int}, optional
            Length of the Fourier transform.

            Returns
            -------
            ceps : ndarray
            The complex cepstrum of the real data sequence `x` computed using the
            Fourier transform.
            ndelay : int
            The amount of samples of circular delay added to `x`.

            The complex cepstrum is given by

            .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}

            where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
            are respectively the forward and backward Fourier transform.

            See Also
            --------
            real_cepstrum: Compute the real cepstrum.
            inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.


            Examples
            --------
            In the following example we use the cepstrum to determine the fundamental
            frequency of a set of harmonics. There is a distinct peak at the quefrency
            corresponding to the fundamental frequency. To be more precise, the peak
            corresponds to the spacing between the harmonics.

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from acoustics.cepstrum import complex_cepstrum

            >>> duration = 5.0
            >>> fs = 8000.0
            >>> samples = int(fs*duration)
            >>> t = np.arange(samples) / fs

            >>> fundamental = 100.0
            >>> harmonics = np.arange(1, 30) * fundamental
            >>> signal = np.sin(2.0*np.pi*harmonics[:,None]*t).sum(axis=0)
            >>> ceps, _ = complex_cepstrum(signal)

            >>> fig = plt.figure()
            >>> ax0 = fig.add_subplot(211)
            >>> ax0.plot(t, signal)
            >>> ax0.set_xlabel('time in seconds')
            >>> ax0.set_xlim(0.0, 0.05)
            >>> ax1 = fig.add_subplot(212)
            >>> ax1.plot(t, ceps)
            >>> ax1.set_xlabel('quefrency in seconds')
            >>> ax1.set_xlim(0.005, 0.015)
            >>> ax1.set_ylim(-5., +10.)

            References
            ----------
            .. [1] Wikipedia, "Cepstrum".
               http://en.wikipedia.org/wiki/Cepstrum
            .. [2] M.P. Norton and D.G. Karczub, D.G.,
               "Fundamentals of Noise and Vibration Analysis for Engineers", 2003.
            .. [3] B. P. Bogert, M. J. R. Healy, and J. W. Tukey:
               "The Quefrency Analysis of Time Series for Echoes: Cepstrum, Pseudo
               Autocovariance, Cross-Cepstrum and Saphe Cracking".
               Proceedings of the Symposium on Time Series Analysis
               Chapter 15, 209-243. New York: Wiley, 1963.

            """

            def _unwrap(phase):
                samples = phase.shape[-1]
                unwrapped = np.unwrap(phase)
                center = (samples + 1) // 2
                if samples == 1:
                    center = 0
                ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
                unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
                return unwrapped, ndelay

            spectrum = np.fft.fft(x, n=n)
            unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
            log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
            ceps = np.fft.ifft(log_spectrum).real

            return ceps, ndelay


        def real_cepstrum(x, n=None):
            r"""Compute the real cepstrum of a real sequence.

            x : ndarray
            Real sequence to compute real cepstrum of.
            n : {None, int}, optional
            Length of the Fourier transform.

            Returns
            -------
            ceps: ndarray
            The real cepstrum.

            The real cepstrum is given by

            .. math:: c[n] = F^{-1}\left{\log_{10}{\left|F{x[n]}\right|}\right}

            where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
            are respectively the forward and backward Fourier transform. Note that
            contrary to the complex cepstrum the magnitude is taken of the spectrum.


            See Also
            --------
            complex_cepstrum: Compute the complex cepstrum of a real sequence.
            inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.

            Examples
            --------
            >>> from acoustics.cepstrum import real_cepstrum


            References
            ----------
            .. [1] Wikipedia, "Cepstrum".
               http://en.wikipedia.org/wiki/Cepstrum

            """
            spectrum = np.fft.fft(x, n=n)
            ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

            return ceps


        def inverse_complex_cepstrum(ceps, ndelay):
            r"""Compute the inverse complex cepstrum of a real sequence.

            ceps : ndarray
            Real sequence to compute inverse complex cepstrum of.
            ndelay: int
            The amount of samples of circular delay added to `x`.

            Returns
            -------
            x : ndarray
            The inverse complex cepstrum of the real sequence `ceps`.

            The inverse complex cepstrum is given by

            .. math:: x[n] = F^{-1}\left{\exp(F(c[n]))\right}

            where :math:`c_[n]` is the input signal and :math:`F` and :math:`F_{-1}
            are respectively the forward and backward Fourier transform.

            See Also
            --------
            complex_cepstrum: Compute the complex cepstrum of a real sequence.
            real_cepstrum: Compute the real cepstrum of a real sequence.

            Examples
            --------
            Taking the complex cepstrum and then the inverse complex cepstrum results
            in the original sequence.

            >>> import numpy as np
            >>> from acoustics.cepstrum import inverse_complex_cepstrum
            >>> x = np.arange(10)
            >>> ceps, ndelay = complex_cepstrum(x)
            >>> y = inverse_complex_cepstrum(ceps, ndelay)
            >>> print(x)
            >>> print(y)

            References
            ----------
            .. [1] Wikipedia, "Cepstrum".
               http://en.wikipedia.org/wiki/Cepstrum

            """

            def _wrap(phase, ndelay):
                ndelay = np.array(ndelay)
                samples = phase.shape[-1]
                center = (samples + 1) // 2
                wrapped = phase + np.pi * ndelay[..., None] * np.arange(samples) / center
                return wrapped

            log_spectrum = np.fft.fft(ceps)
            spectrum = np.exp(log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay))
            x = np.fft.ifft(spectrum).real
            return x


        def minimum_phase(x, n=None):
            r"""Compute the minimum phase reconstruction of a real sequence.

            x : ndarray
            Real sequence to compute the minimum phase reconstruction of.
            n : {None, int}, optional
            Length of the Fourier transform.

            Compute the minimum phase reconstruction of a real sequence using the
            real cepstrum.

            Returns
            -------
            m : ndarray
            The minimum phase reconstruction of the real sequence `x`.

            See Also
            --------
            real_cepstrum: Compute the real cepstrum.

            Examples
            --------
            >>> from acoustics.cepstrum import minimum_phase


            References
            ----------
            .. [1] Soo-Chang Pei, Huei-Shan Lin. Minimum-Phase FIR Filter Design Using
               Real Cepstrum. IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS-II:
               EXPRESS BRIEFS, VOL. 53, NO. 10, OCTOBER 2006

            """
            if n is None:
                n = len(x)
            ceps = real_cepstrum(x, n=n)
            odd = n % 2
            window = np.concatenate(([1.0], 2.0 * np.ones((n + odd) // 2 - 1), np.ones(1 - odd), np.zeros((n + odd) // 2 - 1)))

            m = np.fft.ifft(np.exp(np.fft.fft(window * ceps))).real

            return m

#Ignoring the first 50 observations and taking rest of the observation

        ICA = X[0][50:500]
        sr = len(all_vals)/60
        SP = sr / ((24/60) + 1)
# Calculating the highest peak from cepstrum values
        MAX = np.argmax(np.array(complex_cepstrum(X)[0]))

        rr = (sr * 60) / MAX

        from numpy import fft
        F = np.abs(fft.rfft(ICA))

        # plt.plot(F)
        squarer = lambda t: t ** 2
        vfunc = np.vectorize(squarer)
        # print(vfunc(F))
        # plt.plot(vfunc(F))
        squaredF = vfunc(F)
        logF = np.log(squaredF)

        # logF
        inverseFFT = np.abs(fft.ifft(logF))
        # len(complex_cepstrum(X)[0][0])
        # len(ICA)
        # len(inverseFFT)
        log_X = np.log(np.abs(X[0]))

        windowed_signal = np.hamming(len(all_vals)) * X[0]

        log_X = np.log(np.abs(X)) # np.fft.rfft(windowed_signal)
        sample_freq = len(all_vals)/60


        freq_vector = np.fft.rfftfreq(len(all_vals), d=sample_freq)
        quefrency = np.fft.rfftfreq(X[0].size, freq_vector[1] - freq_vector[0])


        cepstrum = np.fft.rfft(log_X)

        # len(quefrency)
        fig, ax = plt.subplots()
        if len(freq_vector)>100:
            ax.plot(freq_vector, X[0][100:len(freq_vector)+100])
            ax.set_xlabel('frequency (Hz)')
            ax.set_title('Fourier spectrum')
        else:
            ax.plot(freq_vector, X[0][0:len(freq_vector)])
            ax.set_xlabel('frequency (Hz)')
            ax.set_title('Fourier spectrum')
        fig, ax = plt.subplots()
        ax.plot(quefrency, np.abs(cepstrum[0]))
        ax.set_xlabel('quefrency (s)')
        ax.set_title('cepstrum')

        print("Breathing rate: ",np.abs(cepstrum[0])[int(np.floor(SP))])
        # inverseFFT[7]
        #
        time_vector = np.arange(len(all_vals)) / sample_freq

        fig, ax = plt.subplots()
        ax.plot(time_vector, X[0])
        ax.set_xlabel('time (s)')
        ax.set_title('time signal')
        # plt.show()

#Calculating the final breating rate
    if(len(all_vals)>400):
        return {"Len":len(request_data['data']),"bpm":np.abs(cepstrum[0])[int(np.floor(SP))]}
    else:
        return {}
if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
