This code accompanies "Comment on "Accumulating Evidence for the Associate Production of a Neutral Scalar with Mass around 151 GeV"" by Andrew Fowlie. You can reproduce results from the paper by:

    # optional
    pythran methods.py

    # generate pseudo-data
    python3 data.py

    # fit GV coefficient
    python3 gv.py

    # perform MC simulations
    python3 mc.py

    # find resulting significances
    python3 significances.py

    # make plot
    python3 plot.py

In total, the commands took an hour or so to run on my desktop. I find the result:

    Asymptotic local p =  2.269991620571341e-05
    Asymptotic global p =  0.0002702099543116027
    Asymptotic local significances =  4.07812768643986
    Asymptotic global significance =  3.459878133620746
    Asymptotic trials factor =  11.903566156935538
    MC local p =  2e-05
    MC global p =  0.0002
    MC local significances =  4.10747965458625
    MC global significance =  3.5400837992061445
    MC trials factor =  10.0
