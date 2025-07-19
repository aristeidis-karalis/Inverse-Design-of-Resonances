using DSP   # Requires v0.8.0 or higher, otherwise use fs = any value larger than 2*fr (e.g. 4.1*fr) and change line 29 to: p = sort(-fil.p*0.5im*fs, by=real)

## Calculate poles and C of target spectral response
function calc_poles(N, fcenter,fbandwidth, profile,filtertype, pass,stop)
    fl = fcenter*(1 - fbandwidth/2)
    fr = fcenter*(1 + fbandwidth/2)
    if filtertype == "bandpass"
        responsetype = Bandpass(fl, fr)
        t∞ = iseven(N) ? 10^(-stop/20) : 0
        r∞ = sqrt(1-t∞^2) * 1im^(N-1)
    elseif filtertype == "bandstop"
        responsetype = Bandstop(fl, fr)
        t∞ = iseven(N) ? 10^(-pass/20) : 1
        r∞ = sqrt(1-t∞^2) * 1im^(N+1)
    end
    # Note: t∞ is real, r∞ is complex => C∞ = [r∞ t∞; t∞ -conj(r∞)] is unitary & symmetric

    if profile == "butter"
        designmethod = Butterworth(N)
    elseif profile == "cheby"
        designmethod = Chebyshev1(N, pass)
    elseif profile == "icheby"
        designmethod = Chebyshev2(N, stop)
    elseif profile == "ellip"
        designmethod = Elliptic(N, pass, stop)
    end

    fil = analogfilter(responsetype, designmethod)
    p = sort(-fil.p*1im, by=real)

    # output poles in exp(-iωt) convention
    return (conj(p[N+1:end]), t∞, r∞) 
end
