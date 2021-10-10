import numpy as np
import pyyellin as yell
import matplotlib.pyplot as plt
import time
from scipy import integrate

# limit = yell.Limit()
# limit.hello()

m_chi = 10.  # DM mass [GeV/c^2]
recoil_energies = np.arange(0.01, 200.01, 0.01)  # recoil energies [keV]

# signal = yell.SignalModel()
# pdf = signal.pdf(recoil_energies, m_chi)

signal_W = yell.SignalModel(184)  # signal model for A = 184 (tungsten)
signal_I = yell.SignalModel(127)  # signal model for A = 127 (iodine)
pdf_W = signal_W.pdf(recoil_energies, m_chi)
pdf_Ca = signal_I.pdf(recoil_energies, m_chi)


"""
start_time = time.time()
cdf = signal.cdf3(recoil_energies, m_chi)  # TODO: DECIDE WHICH METHOD TO USE
samples = signal.rvs(10000, recoil_energies, m_chi)  # TODO: DECIDE WHICH METHOD TO USE
end_time = time.time()
print(f"Runtime of the third method is {end_time - start_time}")"""

start_time = time.time()


cdf_W = signal_W.cdf2(pdf_W)  # TODO: DECIDE WHICH METHOD TO USE
samples_W = signal_W.rvs2(10000, cdf_W)  # TODO: DECIDE WHICH METHOD TO USE

cdf_Ca = signal_I.cdf2(pdf_Ca)
samples_Ca = signal_I.rvs2(10000, cdf_Ca)


end_time = time.time()
print(f"Runtime of the fourth method is {end_time - start_time}")

ratio_W = 90/100
ratio_Ca = 10/100

# The part below is needed, if one wants to plot only until the first dip
difference = np.shape(pdf_W[1])[0]-np.shape(pdf_Ca[1])[0]
zeros_list = list(np.zeros(abs(difference)))
if difference>0:
    new_list_pdf = list(pdf_Ca[1])+zeros_list
    pdf_Ca = np.array([pdf_W[0], new_list_pdf])  # np.array(new_list_pdf)
    values_to_add_to_cdf = [cdf_Ca[1][-1] for i in range(abs(difference))]
    new_list_cdf = list(cdf_Ca[1])+values_to_add_to_cdf  # list(cdf_Ca[1])+zeros_list
    cdf_Ca = np.array([pdf_W[0], new_list_cdf])  # np.array(new_list_cdf)
else:
    new_list_pdf = list(pdf_W[1])+zeros_list
    pdf_W = np.array([pdf_Ca[0], new_list_pdf])  # np.array(new_list_pdf)
    values_to_add_to_cdf = [cdf_W[1][-1] for i in range(abs(difference))]
    new_list_cdf = list(cdf_W[1])+list(values_to_add_to_cdf)
    cdf_W = np.array([pdf_Ca[0], new_list_cdf])  # np.array(new_list_cdf)


pdf_sum = pdf_W[1]*ratio_W+pdf_Ca[1]*ratio_Ca
cdf_sum = cdf_W[1]*ratio_W+cdf_Ca[1]*ratio_Ca
samples_sum = signal_I.rvs2(10000, np.array([pdf_Ca[0], cdf_sum]))


plt.yscale("log")
plt.plot(pdf_W[0], ratio_W*pdf_W[1], label="ratio_W*log_pdf_W")
plt.plot(pdf_Ca[0], ratio_Ca*pdf_Ca[1], label="ratio_Ca*log_pdf_Ca")
plt.plot(pdf_W[0], pdf_sum, label="log_pdf_sum")
plt.legend()
plt.show()

plt.yscale("linear")
plt.plot(pdf_W[0], pdf_W[1], label="lin_PDF_W")
plt.plot(cdf_W[0], cdf_W[1], label="CDF_W")
plt.hist(samples_W, 1000, density=True, label="Histogram_W")
plt.legend()
plt.show()

plt.plot(pdf_Ca[0], pdf_Ca[1], label="lin_PDF_Ca")
plt.plot(cdf_Ca[0], cdf_Ca[1], label="CDF_Ca")
plt.hist(samples_Ca, 1000, density=True, label="Histogram_Ca")
plt.legend()
plt.show()

plt.plot(pdf_W[0], pdf_sum, label="lin_PDF_sum")
plt.plot(pdf_W[0], cdf_sum, label="CDF_sum")
plt.hist(samples_sum, 1000, density=True, label="Histogram")
plt.legend()
plt.show()

"""plt.yscale("log")
plt.plot(pdf[0], pdf[1], label="log_PDF")
plt.legend()
plt.show()
plt.yscale("linear")
plt.plot(pdf[0], pdf[1], label="lin_PDF")
plt.plot(cdf[0], cdf[1], label="CDF")
plt.hist(samples, 1000, density=True, label="Histogram")
plt.legend()
plt.show()"""
