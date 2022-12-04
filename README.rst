# pyyellin
A Python implementation of Yellin's maximum gap and optimum interval method, for an arbitrary number of data sets and
dimensions.

Motivation:

Let's assume that a researcher wants to discover a signal which is below the sensitivity
of their experiment. Since the signal in this case cannot be directly detected,
only upper limits can be set using the sensitivity of their experiment. Such an experiment
might be contaminated with an unknown background, which might be
hard to remove. But if the distribution of this unknown background were to be
different than the distribution of the expected signal, this difference can be used
to set a stronger upper limit using the optimum interval method proposed by S.
Yellin[1].

The optimum interval method makes use of randomly tabulated data and no
analytical solution exists as for the maximum gap method, also proposed by S.
Yellin, however not as good as the optimum interval method at setting strong upper
limits. Since a lot of values need to be tabulated randomly, using a computer to do
the calculations is a must. The original script for these calculations is written in
Fortran some 20 years ago. Fortran is currently out of fashion and few researchers
can fully understand the script.
This is where the need for an updated version of the implementation arises.
The package, ModeLimit, written in Python is the first-ever open-source implementation
of Yellin's optimum interval method in Python which allows users to
model the expected signal of the dark matter particles depending on their mass,
tabulate data with custom size parameters and apply the optimum interval method
to set upper limits on the cross sections of the dark matter, which is required to
draw the exclusion charts.
Python is currently one of the most widely used programming languages in
the field of data analysis. In comparison to Fortran it is much easier to learn and
code in Python, which is the primary advantage of implementing the method in a
Python package. It is modern and allows many more researchers to make use of
the method.

There are five main use cases, for which there are respective code examples:

1) If you want to utilise the built-in signal model for the calculation of expected number of events and you want to
determine the limits using the maximum gap method, please refer to the example codes in example_sm_mg.py.

2) If you want to utilise the built-in signal model for the calculation of expected number of events and you want to
determine the limits using the optimum interval method, please refer to the example codes in example_sm_oi.py.

3) If you do not want to utilise the built-in signal model for the calculation of expected number of events but you
want to determine the limits using the maximum gap method, please refer to the example code in example_am_mg.py.

4) If you do not want to utilise the built-in signal model for the calculation of expected number of events but you
want to determine the limits using the optimum interval method, please refer to the example codes in example_am_oi.py.

5) If you are just interested in the modelling of the signal, then please refer to the example codes in example_sm.py.

When using optimum interval method, the first run needs to create a table, therefore the first argument of the
set table variables needs to be set to True, instead of False as shown in examples. In the following runs, you may
choose to use the already tabulated data to shorten run time. In order to minimize statistical errors a high number of
datasets per expected number of events (second argument in the method) is advised. Setting the max_mu value of the
set_mu_interval method is also advised for calculating cross sections corresponding to a high number of expected
events.

table_1: 100 arrays per mu, min_mu=0.25, max_mu=50, mu_size=1000
table_2: 1000 arrays per mu, min_mu=0.25, max_mu=250, mu_size=1000

When using already tabulated data (meaning the first argument of set_table_variables is set to False) the values of the
tabulated data must match to the value set in the second argument, the number of arrays per mu, and to the values
set in set_mu_interval method. Otherwise the result may be faulty.

Generally, if you aren't getting any results for some masses, you might consider setting the mu interval larger.

Abbreviations of the example codes:
sm: Signal model, meaning the built-in signal model will be used to determine expected number of events.
am: Another model, meaning another model will be used to determine expected number of events.
mg: Maximum gap method will be used to determine limits.
oi: Optimum interval method will be used to determine limits.

[1] S. Yellin. "Finding an upper limit in the presence of an unknown background".
    In: Physical Review D 66.3 (Aug. 2002). doi: 10.1103/physrevd.
    66.032005. url: https://doi.org/10.1103%2Fphysrevd.66.032005
    (cit. on pp. 1, 21, 24).