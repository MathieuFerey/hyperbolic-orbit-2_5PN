import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst
import sympy as sy

from latex2sympy2 import latex2sympy
from scipy.special import cbrt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

from plot_tools import *


# Utilities =================================================

def derivative(f,x,dx=1e-3): # erivative estimation
    
    f_plus_dx = interp1d(x,f,fill_value='extrapolate')(x+dx)
    f_minus_dx = interp1d(x,f,fill_value='extrapolate')(x-dx)

    return ((f_plus_dx-f_minus_dx)/(2*dx))

def dot(a, b) : # dot product in time, a and b shape (3, len(t)), return a.b shape (len(t))

    adb = np.zeros(len(a[0]))

    for i in range(len(a[0])) :
        adb[i] = np.dot(a[:,i],b[:,i])

    return adb


def cross(a, b) : # cross product in time, a and b shape (3, len(t)), return a.b shape (len(t))

    acb = np.zeros((3,len(a[0])))

    for i in range(len(a[0])) :
        acb[:,i] = np.cross(a[:,i],b[:,i])

    return acb

# Conversion of PN accurate parameters in terms of E, L, L.S1 and L.S2 computed in Mathematica from LaTeX to Python ========================================

def orbit_tex2py(param = 'all') :
    
    if param == 'all' or param == 'coord' :
        print('Radial and angular coordinates:')

        r_N_sy = latex2sympy(r'\frac{1}{\xi^{2 / 3}}(a_t \cosh u-1)')
        print('r_N = ', r_N_sy)

        r_1PN_sy = latex2sympy(r'\frac{\xi^{2 / 3}}{6(a_t \cosh u-1)}((7 \eta-6) a_t \cosh u+2(\eta-9))')
        print('r_1PN = ', r_1PN_sy)

        r_2PN_sy = latex2sympy(r'\frac{\xi^{4 / 3}}{72(a_t^2-1)(a_t \cosh u-1)}((a_t^2-1) a_t(35 \eta^2-231 \eta+72) \cosh u-2 a_t^2(4 \eta^2+15 \eta+36)+8 \eta^2+534 \eta   -216)')
        print('r_2PN = ', r_2PN_sy)

        r_3PN_sy = latex2sympy(r'\frac{\xi^2}{181440(a_t^2-1)^2(a_t \cosh u-1)}(280 a_t^4(16 \eta^3+90 \eta^2-81 \eta+432)+140(a_t^2-1)^2 a_t(49 \eta^3-3933 \eta^2   +7047 \eta-864) \cosh u-a_t^2(8960 \eta^3+3437280 \eta^2+81(1435 \pi^2-134336) \eta+3144960)+4480 \eta^3-761040 \eta^2   -348705 \pi^2 \eta+12143736 \eta-4233600)')
        print('r_3PN = ', r_3PN_sy)

        phi_1PN_sy = latex2sympy(r'\frac{\xi^{2 / 3}}{(a_t^2-1)(a_t \cosh u-1)}(a_t \sqrt{a_t^2-1}(4-\eta) \sinh u+3 \eta(a_t \cosh u-1))')
        print('phi_1PN = ', phi_1PN_sy)

        phi_2PN_sy = latex2sympy(r'\frac{\xi^{4 / 3}}{192(a_t^2-1)^{5 / 2}(a_t \cosh u-1)^2}(a_t(a_t^2-1)(2(a_t^2(384-\eta(7 \eta+275))+4(\eta(\eta+137)-792)) \sinh u   +a_t(a_t^2(\eta(55 \eta-109)+384)-4(\eta(13 \eta+41)-600)) \sinh 2 u)+6 \sqrt{a_t^2-1}(a_t \cosh u-1)^2   \times(a_t^3(1-3 \eta) \eta \sin 3 \nu-8 \nu(a_t^2(26 \eta-51)+28 \eta-78)+4 a_t^2((19-3 \eta) \eta+1) \sin 2 \nu))')
        print('phi_2PN = ', phi_2PN_sy)

        phi_3PN_sy = latex2sympy(r'\frac{\xi^2}{53760(a_t^2-1)^{7 / 2}(a_t \cosh(u)-1)^3}(\sqrt{a_t^2-1}(a_t \cosh(u)-1)^3(2 a_t^2((280 a_t^2(\eta(\eta(93 \eta-781)+886)+24)+\eta(32(35 \eta(9 \eta-395)+36877)-30135 \pi^2)+84000) \sin(2\nu) + a_t\eta ((35 a_t^2(\eta(129 \eta-137)+33)+4(35\eta(51 \eta-727)+28302)-4305 \pi^2) \sin(3\nu) + 35 a_t(3 a_t(5(\eta-1) \eta+1) \sin(5\nu)+4(3 \eta(5 \eta-19)+82) \sin(4\nu)))) + 420 \nu (16(65 a_t^4+320 a_t^2+56) \eta^2 + (123 \pi^2(a_t^2+4)-32(55 a_t^4+870 a_t^2+793))\eta + 96(26 a_t^4+293 a_t^2+190))) + a_t(a_t^2-1)\sinh(u)(-70 a_t^6 \eta(\eta(71 \eta+61)-639)+1680 a_t^4(\eta-4) \cosh^2(u)(3 a_t \eta(3 \eta-1) \cos(3 \nu) + 8(\eta(3 \eta-19)-1) \cos(2\nu)) + a_t^4(\eta(4(70 \eta(125 \eta-507)-462853)-4305 \pi^2)+3933440)+1680 a_t^2(\eta-4)(3 a_t \eta(3 \eta-1) \cos(3\nu) +8(\eta(3 \eta-19)-1) \cos(2\nu))+a_t^2(6 \eta(140 \eta(25 \eta-397)+1435\pi^2-917424)+7947520) + 4 a_t \cosh(u)(-70 a_t^4(\eta(\eta(39 \eta-719)+2279)-3072)+840 a_t^2(\eta-4)(3 a_t(1-3 \eta) \eta \cos(3\nu) + 8((19-3 \eta) \eta+1) \cos(2\nu)) + a_t^2(\eta(8(35(232-53 \eta) \eta+186959)+4305 \pi^2)-2983680)-20(323904-\eta(4(7 \eta(\eta+45)+56013)-861 \pi^2))) + a_t^2(-70 a_t^4 \eta(\eta(71 \eta+61)-639)+a_t^2(\eta(4(35 \eta(229 \eta-1173)-384978)-4305 \pi^2)+3646720)+20(\eta(861 \pi^2-4(14 \eta(9 \eta-25)+54025))+280000)) \cosh(2u) + 40(\eta(1456 \eta+861 \pi^2-253508)+396480)))')
        print('phi_3PN = ', phi_3PN_sy)

    if param == 'all' or param == 'b' :
        print('\nCoefficients of b = c0*xi**(-2/3) + c1 + c2*xi**(2/3) + c3*xi**(4/3):')

        c0 = latex2sympy(r'\sqrt{a_t^2-1}')
        print('c0 = ', c0)

        c1 = c0*latex2sympy(r'(\frac{\eta-1}{a_t^2-1}+\frac{7 \eta-6}{6})')
        print('c1 = ', c1)

        c2 = c0*latex2sympy(r'(1-\frac{7}{24} \eta+\frac{35}{72} \eta^2+\frac{3-16 \eta}{2(a_t^2-1)}+\frac{7-12 \eta-\eta^2}{2(a_t^2-1)^2})')
        print('c2 = ', c2)

        c3 = c0*latex2sympy(r'(-\frac{2}{3}+\frac{87}{16} \eta-\frac{437}{144} \eta^2+\frac{49}{1296} \eta^3+\frac{36-378 \eta+140 \eta^2+3 \eta^3}{24(a_t^2-1)}+\frac{1}{6720(a_t^2-1)^2}(248640 +(-880496+12915 \pi^2) \eta+40880 \eta^2+3920 \eta^3) +\frac{1}{1680(a_t^2-1)^3}(73080+(-228944+4305 \pi^2) \eta+47880 \eta^2+840 \eta^3)))')
        print('c3 = ', c3)

    if param == 'all' or param == 'orbital' :
        print('\nOrbital parameters:')

        n = latex2sympy(r'(2\epsilon )^{3/2}(1-\frac{(2\epsilon )}{8c^2}(-15+\eta)+\frac{(2\epsilon )^2}{128c^4}(555+30\eta+11\eta^2)+\frac{(2\epsilon )^3}{1024c^6}(653+111\eta+7\eta^2+3\eta^3))')
        print('n = ', n)

        et2 = latex2sympy(r'1+2 \epsilon h^2+\frac{(2 \epsilon)}{4 c^2}(8-8 \eta+(17-7 \eta)(2 \epsilon h^2)) +\frac{(2 \epsilon)^2}{8 c^4}(4(3+18 \eta+5 \eta^2)+(2 \epsilon h^2)(112-47 \eta+16 \eta^2) +\frac{16}{(2 \epsilon h^2)}(-4+7 \eta))+\frac{(2 \epsilon)^3}{840 c^6}(-70(42-830 \eta+321 \eta^2+30 \eta^3) -\frac{525}{8}(2 \epsilon h^2)(-528+200 \eta-77 \eta^2+24 \eta^3) -\frac{3}{4(2 \epsilon h^2)}(73920+(-260272+4305 \pi^2) \eta+61040 \eta^2) -\frac{1}{(2 \epsilon h^2)^2}(53760+(-176024+4305 \pi^2) \eta+15120 \eta^2))')
        print('et2 = ', et2)

        ephi2 = latex2sympy(r'1+2\epsilon h^2+\frac{2\epsilon}{4c^{2}}(-24+(-15+\eta)(2\epsilon h^{2}))  +\frac{(2\epsilon)^{2}}{16c^{4}(2\epsilon h^{2})}(-416+91\eta+15\eta^{2}+2(2 \epsilon h^{2})(-20+17\eta+9\eta^{2})+(2\epsilon h^{2})^{2}(160-31\eta+3\eta^{2})) -\frac{(2\epsilon)^{3}}{13440c^{6}(2\epsilon h^{2})^{2}}(2956800+(-5627206+81795\pi^{2})\eta-14490\eta^{2}-7350\eta^{3}  -(2\epsilon h^{2})^{2}(584640+(17482+4305\pi^{2})\eta+7350\eta^{2}-8190\eta^{3}) +420(2\epsilon h^{2})^{3}(744-248\eta+31\eta^{2}+3\eta^{3}) +14(2\epsilon h^{2})(36960+7(-48716+615\pi^{2})\eta-225\eta^{2}+150\eta^{3})))')
        print('ephi2 = ', ephi2)

        er2 = latex2sympy(r'1+2\epsilon h^{2}+\frac{(2\epsilon )}{4c^{2}}(-24+4\eta+5(-3+\eta)(2\epsilon h^{2}))  +\frac{(2\epsilon )^{2}}{8c^{4}}(60+148\eta+2\eta^{2}+(80-45\eta+4\eta^{2})(2\epsilon h^{2})  +\frac{8}{(2\epsilon h^{2})}(-16+28\eta))+\frac{(2\epsilon )^{3}}{6720c^{6}}(2(1680-(90632+4305\pi^{2})\eta+33600\eta^{2})  +4\eta^{3})-\frac{80}{(2\epsilon h^{2})}(1008+(-21130+861\pi^{2})\eta+2268\eta^{2}) -\frac{16}{(2\epsilon h^{2})^{2}}((53760+(-176024+4305\pi^{2})\eta+15120\eta^{2})))')
        print('er2 = ', er2)

        et_to_er = latex2sympy(r'(1+\frac{(2\epsilon )}{2c^2}(8-3\eta)+\frac{(2\epsilon )^2}{c^4}\frac{1}{(2\epsilon h^2)}(8-14\eta+(36-19\eta+6\eta^2)(\epsilon h^2))+\frac{(2\epsilon )^3}{3360c^6}\frac{1}{(2\epsilon h^2)^2}(-420(2\epsilon h^2)^2(10\eta^3-34\eta^2+65\eta-160)+\epsilon h^2(105840\eta^2+(4305\pi^2-354848)\eta+87360)+30240\eta^2+(8610\pi^2-352048)\eta+107520))')
        print('et_to_er = ', et_to_er)

        ephi_to_er = latex2sympy(r'(1-\frac{(2\epsilon )}{2c^{2}}\eta-\frac{(2\epsilon )^{2}}{32c^{4}}\frac{1}{(2\epsilon h^{2})}(160+357\eta-15\eta^{2}-\eta(-1+11\eta)(2\epsilon h^{2}))   +\frac{(2\epsilon )^{3}}{8960c^{6}}\frac{1}{(2\epsilon h^{2})^{2}}(-70(2\epsilon h^{2})^{2}\eta(31\eta^{2}-\eta-1)+5(2\epsilon h^{2})(-1050\eta^{3})  -1854)\eta-412160)')
        print('ephi_to_er = ', ephi_to_er)

        ephi_to_et = latex2sympy(r'-412159+\frac{2 \epsilon (1648636-618239 \eta)}{c^2}+\frac{\epsilon (\eta^2 (19783606 \epsilon h^2+15)+\eta (65945566 \epsilon h^2-184647589)+32 (1648636 \epsilon h^2+3297267))}{16 c^4 h^2}+\frac{\epsilon (-420 \eta^2 (1121072501 \epsilon^2 h^4+692428759 \epsilon h^2-118701792)+\eta (1105113744000 \epsilon^2 h^4+4 (230426832808+1774344495 \pi ^2) \epsilon h^2+7 (2027822280 \pi ^2-82914144187))-26880 (49459080 \epsilon^2 h^4+21020089 \epsilon h^2-6594544)-210 \epsilon \eta^4 h^2 (62 \epsilon h^2+75)+6300 \epsilon \eta^3 h^2 (15387277 \epsilon h^2+3))}{6720 c^6 h^4}')
        print('ephi_to_et = ', ephi_to_et)

    if param == 'all' or param == 'energy' :
        print('\nEnergy:')

        E_2PN = latex2sympy(r'(\frac{5v^{6}}{16}-\frac{35\eta v^{6}}{16}+\frac{65\eta^{2}v^{6}}{16}+\frac{m}{r}(-\frac{3r_d^{4}\eta}{8}+\frac{9r_d^{4}\eta^{2}}{8}+\frac{r_d^{2}\eta v^{2}}{4}-\frac{15r_d^{2}\eta^{2}v^{2}}{4}+\frac{21v^{4}}{8}-\frac{23\eta v^{4}}{8}-\frac{27\eta^{2}v^{4}}{8})+\frac{m^{2}}{r^{2}}(\frac{r_d^{2}}{2}+\frac{69r_d^{2}\eta}{8}+\frac{3r_d^{2}\eta^{2}}{2}+\frac{7v^{2}}{4}-\frac{55\eta v^{2}}{8}+\frac{\eta^{2}v^{2}}{2})+\frac{m^{3}}{r^{3}}(-\frac{1}{2}-\frac{15\eta}{4}))')
        print('E_2PN = ', E_2PN)

        E_3PN = latex2sympy(r'(\frac{35v^{8}}{128}-\frac{413\eta v^{8}}{128}+\frac{833\eta^{2}v^{8}}{64}-\frac{2261\eta^{3}v^{8}}{128}+\frac{m}{r}(\frac{5r_d^{6}\eta}{16}-\frac{25r_d^{6}\eta^{2}}{16}+\frac{25r_d^{6}\eta^{3}}{16}-\frac{9r_d^{4}\eta v^{2}}{16}+\frac{21r_d^{4}\eta^{2}v^{2}}{4}-\frac{165r_d^{4}\eta^{3}v^{2}}{16}-\frac{21r_d^{2}\eta v^{4}}{16}-\frac{75r_d^{2}\eta^{2}v^{4}}{16}+\frac{375r_d^{2}\eta^{3}v^{4}}{16}+\frac{55v^{6}}{16}-\frac{215\eta v^{6}}{16}+\frac{29\eta^{2}v^{6}}{4}+\frac{325\eta^{3}v^{6}}{16})+\frac{m^{2}}{r^{2}}(-\frac{731r_d^{4}\eta}{48}+\frac{41r_d^{4}\eta^{2}}{4}+6r_d^{4}\eta^{3}+\frac{3r_d^{2}v^{2}}{4}+\frac{31r_d^{2}\eta v^{2}}{2}-\frac{815r_d^{2}\eta^{2}v^{2}}{16}-\frac{81r_d^{2}\eta^{3}v^{2}}{4}+\frac{135v^{4}}{16}-\frac{97\eta v^{4}}{8}+\frac{203\eta^{2}v^{4}}{8}-\frac{27\eta^{3}v^{4}}{4})+\frac{m^{3}}{r^{3}}(\frac{3r_d^{2}}{2}+\frac{803r_d^{2}\eta}{840}+\frac{51r_d^{2}\eta^{2}}{4}+\frac{7r_d^{2}\eta^{3}}{2}-\frac{123r_d^{2}\eta\pi^{2}}{64}+\frac{5v^{2}}{4}-\frac{6747\eta v^{2}}{280}-\frac{21\eta^{2}v^{2}}{4}+\frac{\eta^{3}v^{2}}{2}+\frac{41\eta\pi^{2}v^{2}}{64}+22r_d^{2}\eta\ln(\frac{r}{r_{0}})-\frac{22\eta v^{2}}{3}\ln(\frac{r}{r_{0}}))+\frac{m^{4}}{r^{4}}(\frac{3}{8}+\frac{2747\eta}{140}-\frac{11\lambda\eta}{3}-\frac{22\eta}{3}\ln(\frac{r}{r_{0}})))')
        print('E_3PN = ', E_3PN)


    if param == 'all' or param == 'GW' : 
        print('\nGravitational waveforms:')

        h_plus_LO = latex2sympy(r'-dr^2 (\cos (\Theta ) (\cos (\alpha ) \cos (\phi )-\sin (\alpha ) \cos (\iota ) \sin (\phi ))-\sin (\Theta ) \sin (\iota ) \sin (\phi ))^2+dr^2 (\cos (\alpha ) \cos (\iota ) \sin (\phi )+\sin (\alpha ) \cos (\phi ))^2+r (2 dr d\phi (\cos (\Theta ) (\sin (\alpha ) \cos (\iota ) \cos (\phi )+\cos (\alpha ) \sin (\phi ))+\sin (\Theta ) \sin (\iota ) \cos (\phi )) (\cos (\Theta ) (\cos (\alpha ) \cos (\phi )-\sin (\alpha ) \cos (\iota ) \sin (\phi ))-\sin (\Theta ) \sin (\iota ) \sin (\phi ))-2 dr d\phi (\cos (\alpha ) \cos (\iota ) \sin (\phi )+\sin (\alpha ) \cos (\phi )) (\sin (\alpha ) \sin (\phi )-\cos (\alpha ) \cos (\iota ) \cos (\phi )))+r^2 (d\phi^2 (\sin (\alpha ) \sin (\phi )-\cos (\alpha ) \cos (\iota ) \cos (\phi ))^2-d\phi^2 (\cos (\Theta ) (\sin (\alpha ) \cos (\iota ) \cos (\phi )+\cos (\alpha ) \sin (\phi ))+\sin (\Theta ) \sin (\iota ) \cos (\phi ))^2)+z ((\cos (\Theta ) (\cos (\alpha ) \cos (\phi )-\sin (\alpha ) \cos (\iota ) \sin (\phi ))-\sin (\Theta ) \sin (\iota ) \sin (\phi ))^2-(\sin (\alpha ) (-\cos (\phi ))-\cos (\alpha ) \cos (\iota ) \sin (\phi ))^2)')
        print('h_plus_LO = ', h_plus_LO)


def orbit_tex2py_NLOSO(param = 'all') :
    
    if param == 'all' or param == 'orbital' :
        print('\nOrbital parameters:')

        et2 = latex2sympy(r'\left(1 + 2\epsilon L^2\right)+\frac{1}{c^2}\left(-7 \epsilon^2 \eta  L^2+17 \epsilon^2 L^2-4 \epsilon \eta +4 \epsilon\right)-\frac{1}{c^3 L^2}\left(\left(4 \epsilon \sqrt{1-4 \eta }-2 \epsilon \eta+4 \epsilon\right)L\cdot S_1+\left(-2 \epsilon \eta-4 \epsilon \sqrt{1-4 \eta }+4 \epsilon\right)L\cdot S_2\right)+\frac{1}{c^4 L^2}\left(16 \epsilon^3 \eta ^2 L^4-47 \epsilon^3 \eta  L^4+112 \epsilon^3 L^4+10 \epsilon^2 \eta ^2 L^2+2 \epsilon^2 \eta  L^2+4 \epsilon^2 L^2+11 \epsilon \eta -17 \epsilon\right)+\frac{1}{c^5}\frac{1}{2 \left(2 \epsilon L^6+L^4\right)}\left(\left(-90 \epsilon^{5/2} (\eta -2) L^4 \sqrt{\epsilon-4 \epsilon \eta }+\epsilon^{3/2} (236-79 \eta ) L^2 \sqrt{\epsilon-4 \epsilon \eta }+2 \epsilon^3 \left(32 \eta ^2-14 \sqrt{1-4 \eta } \eta -159 \eta +34 \sqrt{1-4 \eta }+124\right) L^4+\epsilon^2 \left(48 \eta ^2-16 \sqrt{1-4 \eta } \eta -315 \eta +16 \sqrt{1-4 \eta }+252\right) L^2+\epsilon \left(8 \eta ^2-78 \eta +64\right)+2 \sqrt{\epsilon} (32-9 \eta ) \sqrt{\epsilon-4 \epsilon \eta }\right) L\cdot S_1+\left(90 \epsilon^{5/2} (\eta -2) L^4 \sqrt{\epsilon-4 \epsilon \eta }+\epsilon^{3/2} (79 \eta -236) L^2 \sqrt{\epsilon-4 \epsilon \eta }+2 \epsilon^3 \left(32 \eta ^2+\left(14 \sqrt{1-4 \eta }-159\right) \eta -34 \sqrt{1-4 \eta }+124\right) L^4+\epsilon^2 \left(48 \eta ^2+\left(16 \sqrt{1-4 \eta }-315\right) \eta -16 \sqrt{1-4 \eta }+252\right) L^2+\epsilon \left(8 \eta ^2-78 \eta +64\right)+2 \sqrt{\epsilon} (9 \eta -32) \sqrt{\epsilon-4 \epsilon \eta }\right) L\cdot S_2\right)')
        print('et2 = ', et2)

        er2 = latex2sympy(r'\left(1 + 2\epsilon L^2\right)+ \frac{1}{c^2}\left(5 \epsilon^2 \eta  L^2-15 \epsilon^2 L^2+2 \epsilon \eta -12 \epsilon\right)+ \frac{1}{c^3}\left(\left(-4 \epsilon^2 \eta +8 \epsilon^2 \sqrt{1-4 \eta }+8 \epsilon^2+\frac{8 \epsilon \sqrt{1-4 \eta }}{L^2}-\frac{4 \epsilon \eta }{L^2}+\frac{8 \epsilon}{L^2}\right)L\cdot S_1 + \left(-4 \epsilon^2 \eta -8 \epsilon^2 \sqrt{1-4 \eta }+8 \epsilon^2-\frac{4 \epsilon \eta }{L^2}-\frac{8 \epsilon \sqrt{1-4 \eta }}{L^2}+\frac{8 \epsilon}{L^2}\right)L\cdot S_2\right)+ \frac{1}{c^4}\left(4 \epsilon^3 \eta ^2 L^2-55 \epsilon^3 \eta  L^2+80 \epsilon^3 L^2+\epsilon^2 \eta ^2+\epsilon^2 \eta +26 \epsilon^2+\frac{22 \epsilon \eta }{L^2}-\frac{34 \epsilon}{L^2}\right)+ \frac{1}{c^5}\left(\frac{\epsilon}{L^4}\left(\epsilon^2 \left(-6 \eta ^2+\left(19 \sqrt{1-4 \eta }+49\right) \eta -80 \left(\sqrt{1-4 \eta }+1\right)\right) L^4+2 \epsilon \left(5 \eta ^2-8 \sqrt{1-4 \eta } \eta -35 \eta +2 \sqrt{1-4 \eta }+2\right) L^2+8 \eta ^2-6 \left(3 \sqrt{1-4 \eta }+13\right) \eta +64 \left(\sqrt{1-4 \eta }+1\right)\right)L\cdot S_1 + \frac{\epsilon}{L^4} \left(\epsilon^2 \left(-6 \eta ^2+\left(49-19 \sqrt{1-4 \eta }\right) \eta +80 \left(\sqrt{1-4 \eta }-1\right)\right) L^4+2 \epsilon \left(5 \eta ^2+\left(8 \sqrt{1-4 \eta }-35\right) \eta -2 \sqrt{1-4 \eta }+2\right) L^2+8 \eta ^2-78 \eta +18 \eta  \sqrt{1-4 \eta }-64 \sqrt{1-4 \eta }+64\right)L\cdot S_2\right)')
        print('er2 = ', er2)

        ar = latex2sympy(r'\frac{1}{2 \epsilon}+\frac{7-\eta }{4 c^2}+\frac{1}{2 c^3 L^2}\left(\left(\eta - 2 \sqrt{1-4 \eta } + 2\right) L\cdot S_1+\left(\eta +2 \sqrt{1-4 \eta }-2\right) L\cdot S_2\right)+\frac{1}{8 c^4 L^2}\left(\epsilon \left(\eta ^2+10 \eta +1\right) L^2-22 \eta +34\right)+\frac{1}{8 c^5 L^4}\left(L\cdot S_1 \left(\epsilon \left(-6 \eta ^2+\left(5 \sqrt{1-4 \eta }+19\right) \eta -8 \left(\sqrt{1-4 \eta }+1\right)\right) L^2-8 \eta ^2-64 \left(\sqrt{1-4 \eta }+1\right)+6 \left(3 \sqrt{1-4 \eta }+13\right) \eta \right)-L\cdot S_2 \left(\epsilon \left(6 \eta ^2+\left(5 \sqrt{1-4 \eta }-19\right) \eta -8 \sqrt{1-4 \eta }+8\right) L^2+8 \eta ^2+18 \sqrt{1-4 \eta } \eta -78 \eta -64 \sqrt{1-4 \eta }+64\right)\right)')
        print('ar = ', ar)

        n = latex2sympy(r'-2 \left(\sqrt{2} \epsilon^{3/2}\right)+\frac{\epsilon^{5/2}}{\sqrt{2} c^2} (\eta -15)-\frac{\epsilon^{7/2}}{8 \sqrt{2} c^4} \left(11 \eta ^2+30 \eta +555\right)')
        print('n = ', n)

        ephi2 = latex2sympy(r'(1+2\epsilon L^2) + \frac{1}{c^2}\left(\epsilon^2 \eta  L^2-15 \epsilon^2 L^2-12 \epsilon\right) + \frac{1}{c^3}\left(\left(-4 \epsilon^2 \eta +8 \epsilon^2 \sqrt{1-4 \eta }+8 \epsilon^2+\frac{8 \epsilon \sqrt{1-4 \eta }}{L^2}-\frac{4 \epsilon \eta }{L^2}+\frac{8 \epsilon}{L^2}\right)L\cdot S_1 + \left(-4 \epsilon^2 \eta -8 \epsilon^2 \sqrt{1-4 \eta }+8 \epsilon^2-\frac{4 \epsilon \eta }{L^2}-\frac{8 \epsilon \sqrt{1-4 \eta }}{L^2}+\frac{8 \epsilon}{L^2}\right)L\cdot S_2\right) + \frac{1}{c^4}\left(\frac{3}{2} \epsilon^3 \eta ^2 L^2-15 \epsilon^3 \eta  L^2+80 \epsilon^3 L^2+\frac{9 \epsilon^2 \eta ^2}{2}+44 \epsilon^2 \eta -8 \epsilon^2+\frac{15 \epsilon \eta ^2}{8 L^2}+\frac{29 \epsilon \eta }{L^2}-\frac{51 \epsilon}{L^2}\right) + \frac{1}{c^5}\left(\left(31 \epsilon^3 \eta +\epsilon^3 \eta  \sqrt{1-4 \eta }-80 \epsilon^3 \sqrt{1-4 \eta }-80 \epsilon^3+\frac{4 \epsilon^2 \eta ^2}{L^2}+\frac{68 \epsilon^2 \sqrt{1-4 \eta }}{L^2}-\frac{30 \epsilon^2 \sqrt{1-4 \eta } \eta }{L^2}-\frac{144 \epsilon^2 \eta }{L^2}+\frac{68 \epsilon^2}{L^2}+\frac{3 \epsilon \eta ^2}{2 L^4}+\frac{96 \epsilon \sqrt{1-4 \eta }}{L^4}-\frac{33 \epsilon \sqrt{1-4 \eta } \eta }{2 L^4}-\frac{213 \epsilon \eta }{2 L^4}+\frac{96 \epsilon}{L^4}\right)L\cdot S_1 + \left(\epsilon^3 \left(-\sqrt{1-4 \eta }\right) \eta +31 \epsilon^3 \eta +80 \epsilon^3 \sqrt{1-4 \eta }-80 \epsilon^3+\frac{4 \epsilon^2 \eta ^2}{L^2}+\frac{30 \epsilon^2 \eta  \sqrt{1-4 \eta }}{L^2}-\frac{144 \epsilon^2 \eta }{L^2}-\frac{68 \epsilon^2 \sqrt{1-4 \eta }}{L^2}+\frac{68 \epsilon^2}{L^2}+\frac{3 \epsilon \eta ^2}{2 L^4}+\frac{33 \epsilon \eta  \sqrt{1-4 \eta }}{2 L^4}-\frac{96 \epsilon \sqrt{1-4 \eta }}{L^4}-\frac{213 \epsilon \eta }{2 L^4}+\frac{96 \epsilon}{L^4}\right)L\cdot S_2\right)')
        print('ephi2 = ', ephi2)

    if param == 'all' or param == 'Kepler' :
        print('\nKepler\'s equation:')

        f_4t = latex2sympy(r'\frac{3 \sqrt{2} \epsilon^{3/2} (5-2 \eta )}{L}')
        print('f_4t = ', f_4t)

        f_5t = latex2sympy(r'\frac{\epsilon^{3/2}}{2 \sqrt{2} L^3} \left(\left(-14 \eta ^2+\left(35 \sqrt{1-4 \eta }+73\right) \eta -48 \left(\sqrt{1-4 \eta }+1\right)\right) L\cdot S_1+\left(-14 \eta ^2+\left(73-35 \sqrt{1-4 \eta }\right) \eta +48 \left(\sqrt{1-4 \eta }-1\right)\right) L\cdot S_2\right)')
        print('f_5t = ', f_5t)

        g_4t = latex2sympy(r'-\frac{\epsilon^{3/2} \eta  (\eta +4) \sqrt{4 \epsilon L^2+2}}{4 L}')
        print('g_4t = ', g_4t)

        g_5t = latex2sympy(r'\frac{\epsilon^{3/2}}{2 L^3} \sqrt{4 \epsilon L^2+2} \left(\left(-3 \eta ^2+\left(9 \sqrt{1-4 \eta }+11\right) \eta -4 \left(\sqrt{1-4 \eta }+1\right)\right) L\cdot S_1+\left(-3 \eta ^2-9 \sqrt{1-4 \eta } \eta +11 \eta +4 \sqrt{1-4 \eta }-4\right) L\cdot S_2\right)')
        print('g_5t = ', g_5t)

    if param == 'all' or param == 'angular' :
        print('\nAngular equation:')

        d2 = latex2sympy(r'L+\frac{1}{c^2}\left(3 \epsilon \eta  L-\epsilon L\right) + \frac{1}{c^4}\left(3 \epsilon^2 \eta ^2 L-\frac{9}{2} \epsilon^2 \eta  L+\epsilon^2 L\right)')
        print('d2 = ', d2)

        d3 = latex2sympy(r'\frac{1}{c^2}\left(2 \eta  L-4 L\right) + \frac{1}{2 L c^3}\left(\left(-\eta +2 \sqrt{1-4 \eta }+2\right)L\cdot S_1 + \left(\eta +2 \sqrt{1-4 \eta }-2\right)L\cdot S_2\right) + \frac{1}{c^4}\left(8 \epsilon \eta ^2 L-22 \epsilon \eta  L+4 \epsilon L\right) + \frac{\eta \epsilon}{8 L c^5}\left(\left(-18 \eta +31 \sqrt{1-4 \eta }+21\right)L\cdot S_1 + \left(-18 \eta -31 \sqrt{1-4 \eta }+21\right)L\cdot S_2\right)')
        print('d3 = ', d3)

        d4 = latex2sympy(r'\frac{1}{c^4}\left(5 \eta ^2 L-11 \eta  L+\frac{17 L}{2}\right) + \frac{1}{8 L c^5}\left(\left(-18 \eta ^2+23 \sqrt{1-4 \eta } \eta +21 \eta -24 \sqrt{1-4 \eta }-24\right)L\cdot S_1 + \left(-18 \eta ^2-23 \sqrt{1-4 \eta } \eta +21 \eta +24 \sqrt{1-4 \eta }-24\right)L\cdot S_2\right)')
        print('d4 = ', d4)

        d5 = latex2sympy(r'\frac{1}{c^4}\left(\frac{\eta  L^3}{2}-\eta ^2 L^3\right) + \frac{1}{c^5}\left(\left(\frac{9 \eta ^2 L}{4}-\frac{15}{4} \sqrt{1-4 \eta } \eta  L-\frac{17 \eta  L}{4}+\sqrt{1-4 \eta } L+L\right)L\cdot S_1 + \left(\frac{9 \eta ^2 L}{4}+\frac{15}{4} \sqrt{1-4 \eta } \eta  L-\frac{17 \eta  L}{4}-\sqrt{1-4 \eta } L+L\right)L\cdot S_2\right)')
        print('d5 = ', d5)

    if param == 'all' or param == 'precession' :
        print('\nPrecession equations:')

        f_3L = latex2sympy(r'\left(1 + \sqrt{1 - 4\eta} -\frac{\eta}{2}\right)\frac{1}{r^3}')
        print('f_3L = ', f_3L)

        f_5L = latex2sympy(r'\left(\epsilon \eta  \left(-18 \eta +31 \sqrt{1-4 \eta }+21\right) r^2+6 \eta  \left(\eta -\sqrt{1-4 \eta }-1\right) L^2+\left(-18 \eta ^2+23\eta\sqrt{1-4 \eta }+21\eta -24\sqrt{1-4 \eta }-24\right) r\right)\frac{1}{8 r^5}')
        print('f_5L = ', f_5L)

        g_3L = latex2sympy(r'\left(1 - \sqrt{1 - 4\eta} -\frac{\eta}{2}\right)\frac{1}{r^3}')
        print('g_3L = ', g_3L)

        g_5L = latex2sympy(r'\left(\epsilon \eta  \left(-18 \eta -31 \sqrt{1-4 \eta }+21\right) r^2+6 \eta  \left(\eta +\sqrt{1-4 \eta }-1\right) L^2+\left(-18 \eta ^2-23\eta\sqrt{1-4 \eta } +21 \eta +24 \sqrt{1-4 \eta }-24\right) r\right)\frac{1}{8 r^5}')
        print('g_5L = ', g_5L)

    if param == 'all' or param == 'iterative' :
        print('E and L in terms of et and n:')

        E = latex2sympy(r'-\frac{\left(\eta ^2+15 \eta -15\right) n^2}{48 c^4}+\frac{(\eta -15) n^{4/3}}{24 c^2}+\frac{n^{2/3}}{2}')
        print('E = ', E)

        L = latex2sympy(r'\frac{n s_1\text{kds1} \left(\text{et}^2^2 \left(135 (\eta -2) \sqrt{(1-4 \eta ) n^{2/3}}+\left(-94 \eta ^2+38 \sqrt{1-4 \eta } \eta +443 \eta -42 \sqrt{1-4 \eta }-312\right) \sqrt[3]{n}\right)-\text{et}^2 \left(3 (11 \eta +56) \sqrt{(1-4 \eta ) n^{2/3}}+\left(2 \eta ^2-64 \sqrt{1-4 \eta } \eta -169 \eta +168\right) \sqrt[3]{n}\right)+6 (\eta +9) \left(\sqrt{(1-4 \eta ) n^{2/3}}-\sqrt{1-4 \eta } \sqrt[3]{n}\right)\right)-n s_2\text{kds2} \left(\text{et}^2^2 \left(135 (\eta -2) \sqrt{(1-4 \eta ) n^{2/3}}+\left(94 \eta ^2+38 \sqrt{1-4 \eta } \eta -443 \eta -42 \sqrt{1-4 \eta }+312\right) \sqrt[3]{n}\right)-3 \text{et}^2 (11 \eta +56) \sqrt{(1-4 \eta ) n^{2/3}}+\text{et}^2 \left(2 \eta ^2+\left(64 \sqrt{1-4 \eta }-169\right) \eta +168\right) \sqrt[3]{n}+6 (\eta +9) \left(\sqrt{(1-4 \eta ) n^{2/3}}-\sqrt{1-4 \eta } \sqrt[3]{n}\right)\right)}{48 c^5 (\text{et}^2-1)^2 \text{et}^2}+\frac{n \left(\text{et}^2^2 \left(5 \eta ^2-73 \eta +33\right)-2 \text{et}^2 \left(9 \eta ^2-26 \eta +6\right)+\eta ^2-21 \eta +69\right)}{24 c^4 (\text{et}^2-1)^{3/2}}+\frac{n^{2/3} \left(\left(\eta -2 \left(\sqrt{1-4 \eta }+1\right)\right) s_1\text{kds1}+\left(\eta +2 \sqrt{1-4 \eta }-2\right) s_2\text{kds2}\right)}{2 c^3 (\text{et}^2-1)}+\frac{\sqrt[3]{n} (\text{et}^2 (5 \eta -9)+\eta +3)}{6 c^2 \sqrt{\text{et}^2-1}}+\frac{\sqrt{\text{et}^2-1}}{\sqrt[3]{n}}')
        print('L = ', L)



# Mikkola's method ==============================================================================================

def cubic(e, l) : # returns the root of the depressed cubic equation (4e+1/2)s^3 + 3(e-1)s == l

    alpha = (e-1)/(4*e+0.5)
    beta = 0.5*l/(4*e+0.5)

    sign = beta/np.abs(beta)

    z = cbrt(beta + sign*np.sqrt(beta**2+alpha**3))

    return z - alpha/z


def mikkola(e, l, DB_corr = True) :

    s = cubic(e, l)
    ds = 0.0071*s**5/((1+0.45*s**2)*(1+4*s**2)*e) # error on s
    s += ds
    u = 3*np.log(s+np.sqrt(1+s**2))

    # improve result with Danby Burkardt's method

    if DB_corr :

        eshu = e*np.sinh(u)
        echu = e*np.cosh(u)
        
        fu  = -u + eshu - l
        f1u = -1 + echu
        f2u = eshu
        f3u = echu
        f4u = eshu
        f5u = echu

        u1 = -fu/ f1u
        u2 = -fu/(f1u + f2u*u1/2)
        u3 = -fu/(f1u + f2u*u2/2 + f3u*(u2*u2)/6.0)
        u4 = -fu/(f1u + f2u*u3/2 + f3u*(u3*u3)/6.0 + f4u*(u3*u3*u3)/24.0)
        u5 = -fu/(f1u + f2u*u4/2 + f3u*(u4*u4)/6.0 + f4u*(u4*u4*u4)/24.0 + f5u*(u4*u4*u4*u4)/120.0)
        u += u5

    return u


# Spinning compact binaries at 2.5PN =====================================================================================

def PN_param(PN = 5) :

    if PN == 5 :
        PN5 = 1
        PN4 = 1
        PN3 = 1
        PN2 = 1

    elif PN == 4 :
        PN5 = 0
        PN4 = 1
        PN3 = 0
        PN2 = 1
    
    elif PN == 3 :
        PN5 = 0
        PN4 = 0
        PN3 = 1
        PN2 = 1
    
    elif PN == 2 :
        PN5 = 0
        PN4 = 0
        PN3 = 0
        PN2 = 1

    elif PN == 0 :
        PN5 = 0
        PN4 = 0
        PN3 = 0
        PN2 = 0

    return PN2, PN3, PN4, PN5


def spinning_orbit_2_5PN_param(n, et, kds1, kds2, eta, S1, S2, t, PN=5) :

    c = 1
    PN2, PN3, PN4, PN5 = PN_param(PN)

    E =  n**(2/3)/2 - PN4*n**2*(eta**2 + 15*eta - 15)/(48*c**4) + PN2*(n**(4/3)*(eta - 15))/((24*c**2))
    L =  (kds1*n*S1*(-et**2*(n**(1/3)*(2*eta**2 - 1*64*np.sqrt(-1*4*eta + 1)*eta - 1*169*eta + 168) + 3*np.sqrt(n**(2/3)*(-1*4*eta + 1))*(11*eta + 56)) + (n**(1/3)*(-1*94*eta**2 + 38*eta*np.sqrt(-1*4*eta + 1) + 443*eta - 1*42*np.sqrt(-1*4*eta + 1) - 312) + 135*np.sqrt(n**(2/3)*(-1*4*eta + 1))*(eta - 2))*(et**2)**2 + 6*(eta + 9)*(-n**(1/3)*np.sqrt(-1*4*eta + 1) + np.sqrt(n**(2/3)*(-1*4*eta + 1)))) - kds2*n*S2*(et**2*n**(1/3)*(2*eta**2 + eta*(64*np.sqrt(-1*4*eta + 1) - 169) + 168) - 1*3*et**2*(11*eta + 56)*np.sqrt(n**(2/3)*(-1*4*eta + 1)) + (n**(1/3)*(94*eta**2 + 38*eta*np.sqrt(-1*4*eta + 1) - 1*443*eta - 1*42*np.sqrt(-1*4*eta + 1) + 312) + 135*np.sqrt(n**(2/3)*(-1*4*eta + 1))*(eta - 2))*(et**2)**2 + 6*(eta + 9)*(-n**(1/3)*np.sqrt(-1*4*eta + 1) + np.sqrt(n**(2/3)*(-1*4*eta + 1)))))/((48*et**2*c**5*(et**2 - 1)**2))*PN5 + (n*(-1*2*et**2*(9*eta**2 - 1*26*eta + 6) + (5*eta**2 - 1*73*eta + 33)*(et**2)**2 + eta**2 - 1*21*eta + 69))/((24*c**4*(et**2 - 1)**(3/2)))*PN4 + (n**(2/3)*(kds1*S1*(eta - 1*2*(np.sqrt(-1*4*eta + 1) + 1)) + kds2*S2*(eta + 2*np.sqrt(-1*4*eta + 1) - 2)))/((2*c**3*(et**2 - 1)))*PN3 + (n**(1/3)*(et**2*(5*eta - 9) + eta + 3))/((6*c**2*np.sqrt(et**2 - 1)))*PN2 + np.sqrt(et**2 - 1)/(n**(1/3))


    # orbital param
    er2 =  (2*L**2*E + 1) + 1*((L*S1)*kds1*E*(L**4*E**2*(-1*6*eta**2 + eta*(19*np.sqrt(-1*4*eta + 1) + 49) - 1*80*(np.sqrt(-1*4*eta + 1) + 1)) + 2*L**2*E*(5*eta**2 - 1*8*np.sqrt(-1*4*eta + 1)*eta - 1*35*eta + 2*np.sqrt(-1*4*eta + 1) + 2) + 8*eta**2 - 1*6*(3*np.sqrt(-1*4*eta + 1) + 13)*eta + 64*(np.sqrt(-1*4*eta + 1) + 1))/(L**4) + (L*S2)*kds2*E*(L**4*E**2*(-1*6*eta**2 + eta*(-1*19*np.sqrt(-1*4*eta + 1) + 49) + 80*(np.sqrt(-1*4*eta + 1) - 1)) + 2*L**2*E*(5*eta**2 + eta*(8*np.sqrt(-1*4*eta + 1) - 35) - 1*2*np.sqrt(-1*4*eta + 1) + 2) + 8*eta**2 + 18*eta*np.sqrt(-1*4*eta + 1) - 1*78*eta - 1*64*np.sqrt(-1*4*eta + 1) + 64)/(L**4))/c**5*PN5 + 1*(4*L**2*E**3*eta**2 - 1*55*E**3*eta*L**2 + 80*L**2*E**3 + E**2*eta**2 + E**2*eta + 26*E**2 - 34*E/(L**2) + (22*E*eta)/(L**2))/c**4*PN4 + 1*((L*S1)*kds1*(-1*4*E**2*eta + 8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) + (8*E*np.sqrt(-1*4*eta + 1))/(L**2)) + (L*S2)*kds2*(-1*4*E**2*eta - 1*8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) - 8*E*np.sqrt(-1*4*eta + 1)/(L**2)))/c**3*PN3 + 1*(5*L**2*E**2*eta - 1*15*E**2*L**2 + 2*E*eta - 1*12*E)/c**2*PN2
    ar =  1*((L*S1)*kds1*(L**2*E*(-1*6*eta**2 + eta*(5*np.sqrt(-1*4*eta + 1) + 19) - 1*8*(np.sqrt(-1*4*eta + 1) + 1)) - 1*8*eta**2 + 6*eta*(3*np.sqrt(-1*4*eta + 1) + 13) - 1*64*(np.sqrt(-1*4*eta + 1) + 1)) - (L*S2)*kds2*(L**2*E*(6*eta**2 + eta*(5*np.sqrt(-1*4*eta + 1) - 19) - 1*8*np.sqrt(-1*4*eta + 1) + 8) + 8*eta**2 + 18*eta*np.sqrt(-1*4*eta + 1) - 1*78*eta - 1*64*np.sqrt(-1*4*eta + 1) + 64))/(8*L**4*c**5)*PN5 + 1*(L**2*E*(eta**2 + 10*eta + 1) - 1*22*eta + 34)/(8*L**2*c**4)*PN4 + 1*((L*S1)*kds1*(eta - 1*2*np.sqrt(-1*4*eta + 1) - 2) + (L*S2)*kds2*(eta + 2*np.sqrt(-1*4*eta + 1) - 2))/(2*L**2*c**3)*PN3 + 1/(2*E) + (7 - eta)/((4*c**2))*PN2
    ephi2 =  (2*L**2*E + 1) + 1*((L*S1)*kds1*(E**3*eta*np.sqrt(-1*4*eta + 1) + 31*E**3*eta - 1*80*E**3*np.sqrt(-1*4*eta + 1) - 1*80*E**3 - 213*E*eta/(2*L**4) + (3*E*eta**2)/((2*L**4)) - 33*E*eta*np.sqrt(-1*4*eta + 1)/(2*L**4) + (96*E)/(L**4) + (96*E*np.sqrt(-1*4*eta + 1))/(L**4) + (68*E**2)/(L**2) - 144*E**2*eta/(L**2) + (4*E**2*eta**2)/(L**2) + (68*E**2*np.sqrt(-1*4*eta + 1))/(L**2) - 30*E**2*eta*np.sqrt(-1*4*eta + 1)/(L**2)) + (L*S2)*kds2*(E**3*(-1)*np.sqrt(-1*4*eta + 1)*eta + 31*E**3*eta + 80*E**3*np.sqrt(-1*4*eta + 1) - 1*80*E**3 - 213*E*eta/(2*L**4) + (3*E*eta**2)/((2*L**4)) + (33*E*eta*np.sqrt(-1*4*eta + 1))/((2*L**4)) + (96*E)/(L**4) - 96*E*np.sqrt(-1*4*eta + 1)/(L**4) + (68*E**2)/(L**2) - 144*E**2*eta/(L**2) + (4*E**2*eta**2)/(L**2) - 68*E**2*np.sqrt(-1*4*eta + 1)/(L**2) + (30*E**2*eta*np.sqrt(-1*4*eta + 1))/(L**2)))/c**5*PN5 + 1*(3*E**3*eta**2*L**2/2 - 1*15*E**3*eta*L**2 + 80*L**2*E**3 + 44*E**2*eta - 1*8*E**2 + (9*E**2*eta**2)/2 + (15*E*eta**2)/((8*L**2)) - 51*E/(L**2) + (29*E*eta)/(L**2))/c**4*PN4 + 1*((L*S1)*kds1*(-1*4*E**2*eta + 8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) + (8*E*np.sqrt(-1*4*eta + 1))/(L**2)) + (L*S2)*kds2*(-1*4*E**2*eta - 1*8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) - 8*E*np.sqrt(-1*4*eta + 1)/(L**2)))/c**3*PN3 + 1*(L**2*E**2*eta - 1*15*E**2*L**2 - 1*12*E)/c**2*PN2
    
    # Kepler's equation:
    f_4t =  PN4*(3*np.sqrt(2)*E**(3/2)*(-1*2*eta + 5))/L
    f_5t =  PN5*E**(3/2)*((L*S1)*kds1*(-1*14*eta**2 + eta*(35*np.sqrt(-1*4*eta + 1) + 73) - 1*48*(np.sqrt(-1*4*eta + 1) + 1)) + (L*S2)*kds2*(-1*14*eta**2 + eta*(-1*35*np.sqrt(-1*4*eta + 1) + 73) + 48*(np.sqrt(-1*4*eta + 1) - 1)))/((2*np.sqrt(2)*L**3))
    g_4t =  -E**(3/2)*PN4*eta*(eta + 4)*np.sqrt(4*L**2*E + 2)/(4*L)
    g_5t =  PN5*E**(3/2)*np.sqrt(4*L**2*E + 2)*((L*S1)*kds1*(-1*3*eta**2 + eta*(9*np.sqrt(-1*4*eta + 1) + 11) - 1*4*(np.sqrt(-1*4*eta + 1) + 1)) + (L*S2)*kds2*(-1*3*eta**2 - 1*9*np.sqrt(-1*4*eta + 1)*eta + 11*eta + 4*np.sqrt(-1*4*eta + 1) - 4))/((2*L**3))

    # angular equation
    d2 =  L + 1*(3*L*E**2*eta**2 - 1*9*E**2*eta*L/2 + L*E**2)/c**4*PN4 + 1*(3*L*E*eta - L*E)/c**2*PN2
    d3 =  (E*eta)*((L*S1)*kds1*(-1*18*eta + 31*np.sqrt(-1*4*eta + 1) + 21) + (L*S2)*kds2*(-1*18*eta - 1*31*np.sqrt(-1*4*eta + 1) + 21))/((8*L*c**5))*PN5 + 1*((L*S1)*kds1*(-eta + 2*np.sqrt(-1*4*eta + 1) + 2) + (L*S2)*kds2*(-eta - 2*np.sqrt(-1*4*eta + 1) + 2))/(2*L*c**3)*PN3 + 1*(8*L*E*eta**2 - 1*22*E*eta*L + 4*L*E)/c**4*PN4 + 1*(2*L*eta - 1*4*L)/c**2*PN2
    d4 =  1*((L*S1)*kds1*(-1*18*eta**2 + 23*eta*np.sqrt(-1*4*eta + 1) + 21*eta - 1*24*np.sqrt(-1*4*eta + 1) - 24) + (L*S2)*kds2*(-1*18*eta**2 - 1*23*np.sqrt(-1*4*eta + 1)*eta + 21*eta + 24*np.sqrt(-1*4*eta + 1) - 24))/(8*L*c**5)*PN5 + 1*(5*L*eta**2 - 1*11*eta*L + (17*L)/2)/c**4*PN4
    d5 =  1*((L*S1)*kds1*(-1*15*np.sqrt(-1*4*eta + 1)*eta*L/4 + L*np.sqrt(-1*4*eta + 1) + L - 1*17*L*eta/4 + (9*L*eta**2)/4) + (L*S2)*kds2*(15*np.sqrt(-1*4*eta + 1)*eta*L/4 - L*np.sqrt(-1*4*eta + 1) + L - 1*17*L*eta/4 + (9*L*eta**2)/4))/c**5*PN5 + 1*(-L**3*eta**2 + (L**3*eta)/2)/c**4*PN4

    return E, L, ar, np.sqrt(er2), np.sqrt(ephi2), d2, d3, d4, d5, f_4t, f_5t, g_4t, g_5t


def spinning_orbit_2_5PN_param_from_E_L(E, L, kds1, kds2, eta, S1, S2, PN=5) :

    PN2, PN3, PN4, PN5 = PN_param(PN)
    PN0 = 1
    
    # orbital param
    n = PN0*(2*E)**(3/2) - PN2*E**(5/2)*(eta - 15)/np.sqrt(2) + PN4*E**(7/2)*(11*eta**2 + 30*eta + 555)/(8*np.sqrt(2))
    et2 = PN0*(2*L**2*E + 1) + 1*(16*L**4*E**3*eta**2 - 1*47*E**3*eta*L**4 + 112*L**4*E**3 + 10*L**2*E**2*eta**2 + 2*L**2*E**2*eta + 4*L**2*E**2 + 11*E*eta - 1*17*E)/(L**2*c**4)*PN4 + 2*E*((L*S1)*kds1*(-eta + 2*np.sqrt(-1*4*eta + 1) + 2) - (L*S2)*kds2*(eta + 2*np.sqrt(-1*4*eta + 1) - 2))/(L**2*c**3)*PN3 + 1*1*((L*S1)*kds1*(2*L**4*E**3*(32*eta**2 - 1*14*np.sqrt(-1*4*eta + 1)*eta - 1*159*eta + 34*np.sqrt(-1*4*eta + 1) + 124) - 1*90*E**(5/2)*(eta - 2)*L**4*np.sqrt(-1*4*E*eta + E) + L**2*E**2*(48*eta**2 - 1*16*np.sqrt(-1*4*eta + 1)*eta - 1*315*eta + 16*np.sqrt(-1*4*eta + 1) + 252) + L**2*E**(3/2)*(-1*79*eta + 236)*np.sqrt(-1*4*E*eta + E) + 2*np.sqrt(E)*(-1*9*eta + 32)*np.sqrt(-1*4*E*eta + E) + E*(8*eta**2 - 1*78*eta + 64)) + (L*S2)*kds2*(2*L**4*E**3*(32*eta**2 + eta*(14*np.sqrt(-1*4*eta + 1) - 159) - 1*34*np.sqrt(-1*4*eta + 1) + 124) + 90*L**4*E**(5/2)*(eta - 2)*np.sqrt(-1*4*E*eta + E) + L**2*E**2*(48*eta**2 + eta*(16*np.sqrt(-1*4*eta + 1) - 315) - 1*16*np.sqrt(-1*4*eta + 1) + 252) + L**2*E**(3/2)*(79*eta - 236)*np.sqrt(-1*4*E*eta + E) + 2*np.sqrt(E)*(9*eta - 32)*np.sqrt(-1*4*E*eta + E) + E*(8*eta**2 - 1*78*eta + 64)))/(c**5*(2*(2*L**6*E + L**4)))*PN5 + 1*(-1*7*E**2*eta*L**2 + 17*L**2*E**2 - 1*4*E*eta + 4*E)/c**2*PN2
    er2 =  PN0*(2*L**2*E + 1) + 1*((L*S1)*kds1*E*(L**4*E**2*(-1*6*eta**2 + eta*(19*np.sqrt(-1*4*eta + 1) + 49) - 1*80*(np.sqrt(-1*4*eta + 1) + 1)) + 2*L**2*E*(5*eta**2 - 1*8*np.sqrt(-1*4*eta + 1)*eta - 1*35*eta + 2*np.sqrt(-1*4*eta + 1) + 2) + 8*eta**2 - 1*6*(3*np.sqrt(-1*4*eta + 1) + 13)*eta + 64*(np.sqrt(-1*4*eta + 1) + 1))/(L**4) + (L*S2)*kds2*E*(L**4*E**2*(-1*6*eta**2 + eta*(-1*19*np.sqrt(-1*4*eta + 1) + 49) + 80*(np.sqrt(-1*4*eta + 1) - 1)) + 2*L**2*E*(5*eta**2 + eta*(8*np.sqrt(-1*4*eta + 1) - 35) - 1*2*np.sqrt(-1*4*eta + 1) + 2) + 8*eta**2 + 18*eta*np.sqrt(-1*4*eta + 1) - 1*78*eta - 1*64*np.sqrt(-1*4*eta + 1) + 64)/(L**4))/c**5*PN5 + 1*(4*L**2*E**3*eta**2 - 1*55*E**3*eta*L**2 + 80*L**2*E**3 + E**2*eta**2 + E**2*eta + 26*E**2 - 34*E/(L**2) + (22*E*eta)/(L**2))/c**4*PN4 + 1*((L*S1)*kds1*(-1*4*E**2*eta + 8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) + (8*E*np.sqrt(-1*4*eta + 1))/(L**2)) + (L*S2)*kds2*(-1*4*E**2*eta - 1*8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) - 8*E*np.sqrt(-1*4*eta + 1)/(L**2)))/c**3*PN3 + 1*(5*L**2*E**2*eta - 1*15*E**2*L**2 + 2*E*eta - 1*12*E)/c**2*PN2
    ar =  1*((L*S1)*kds1*(L**2*E*(-1*6*eta**2 + eta*(5*np.sqrt(-1*4*eta + 1) + 19) - 1*8*(np.sqrt(-1*4*eta + 1) + 1)) - 1*8*eta**2 + 6*eta*(3*np.sqrt(-1*4*eta + 1) + 13) - 1*64*(np.sqrt(-1*4*eta + 1) + 1)) - (L*S2)*kds2*(L**2*E*(6*eta**2 + eta*(5*np.sqrt(-1*4*eta + 1) - 19) - 1*8*np.sqrt(-1*4*eta + 1) + 8) + 8*eta**2 + 18*eta*np.sqrt(-1*4*eta + 1) - 1*78*eta - 1*64*np.sqrt(-1*4*eta + 1) + 64))/(8*L**4*c**5)*PN5 + 1*(L**2*E*(eta**2 + 10*eta + 1) - 1*22*eta + 34)/(8*L**2*c**4)*PN4 + 1*((L*S1)*kds1*(eta - 1*2*np.sqrt(-1*4*eta + 1) - 2) + (L*S2)*kds2*(eta + 2*np.sqrt(-1*4*eta + 1) - 2))/(2*L**2*c**3)*PN3 + PN0/(2*E) + (7 - eta)/((4*c**2))*PN2
    ephi2 =  PN0*(2*L**2*E + 1) + 1*((L*S1)*kds1*(E**3*eta*np.sqrt(-1*4*eta + 1) + 31*E**3*eta - 1*80*E**3*np.sqrt(-1*4*eta + 1) - 1*80*E**3 - 213*E*eta/(2*L**4) + (3*E*eta**2)/((2*L**4)) - 33*E*eta*np.sqrt(-1*4*eta + 1)/(2*L**4) + (96*E)/(L**4) + (96*E*np.sqrt(-1*4*eta + 1))/(L**4) + (68*E**2)/(L**2) - 144*E**2*eta/(L**2) + (4*E**2*eta**2)/(L**2) + (68*E**2*np.sqrt(-1*4*eta + 1))/(L**2) - 30*E**2*eta*np.sqrt(-1*4*eta + 1)/(L**2)) + (L*S2)*kds2*(E**3*(-1)*np.sqrt(-1*4*eta + 1)*eta + 31*E**3*eta + 80*E**3*np.sqrt(-1*4*eta + 1) - 1*80*E**3 - 213*E*eta/(2*L**4) + (3*E*eta**2)/((2*L**4)) + (33*E*eta*np.sqrt(-1*4*eta + 1))/((2*L**4)) + (96*E)/(L**4) - 96*E*np.sqrt(-1*4*eta + 1)/(L**4) + (68*E**2)/(L**2) - 144*E**2*eta/(L**2) + (4*E**2*eta**2)/(L**2) - 68*E**2*np.sqrt(-1*4*eta + 1)/(L**2) + (30*E**2*eta*np.sqrt(-1*4*eta + 1))/(L**2)))/c**5*PN5 + 1*(3*E**3*eta**2*L**2/2 - 1*15*E**3*eta*L**2 + 80*L**2*E**3 + 44*E**2*eta - 1*8*E**2 + (9*E**2*eta**2)/2 + (15*E*eta**2)/((8*L**2)) - 51*E/(L**2) + (29*E*eta)/(L**2))/c**4*PN4 + 1*((L*S1)*kds1*(-1*4*E**2*eta + 8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) + (8*E*np.sqrt(-1*4*eta + 1))/(L**2)) + (L*S2)*kds2*(-1*4*E**2*eta - 1*8*E**2*np.sqrt(-1*4*eta + 1) + 8*E**2 + (8*E)/(L**2) - 4*E*eta/(L**2) - 8*E*np.sqrt(-1*4*eta + 1)/(L**2)))/c**3*PN3 + 1*(L**2*E**2*eta - 1*15*E**2*L**2 - 1*12*E)/c**2*PN2
    
    # Kepler's equation:
    f_4t =  PN4*(3*np.sqrt(2)*E**(3/2)*(-1*2*eta + 5))/L
    f_5t =  PN5*E**(3/2)*((L*S1)*kds1*(-1*14*eta**2 + eta*(35*np.sqrt(-1*4*eta + 1) + 73) - 1*48*(np.sqrt(-1*4*eta + 1) + 1)) + (L*S2)*kds2*(-1*14*eta**2 + eta*(-1*35*np.sqrt(-1*4*eta + 1) + 73) + 48*(np.sqrt(-1*4*eta + 1) - 1)))/((2*np.sqrt(2)*L**3))
    g_4t =  -E**(3/2)*PN4*eta*(eta + 4)*np.sqrt(4*L**2*E + 2)/(4*L)
    g_5t =  PN5*E**(3/2)*np.sqrt(4*L**2*E + 2)*((L*S1)*kds1*(-1*3*eta**2 + eta*(9*np.sqrt(-1*4*eta + 1) + 11) - 1*4*(np.sqrt(-1*4*eta + 1) + 1)) + (L*S2)*kds2*(-1*3*eta**2 - 1*9*np.sqrt(-1*4*eta + 1)*eta + 11*eta + 4*np.sqrt(-1*4*eta + 1) - 4))/((2*L**3))

    # angular equation
    d2 =  PN0*L + 1*(3*L*E**2*eta**2 - 1*9*E**2*eta*L/2 + L*E**2)/c**4*PN4 + 1*(3*L*E*eta - L*E)/c**2*PN2
    d3 =  (E*eta)*((L*S1)*kds1*(-1*18*eta + 31*np.sqrt(-1*4*eta + 1) + 21) + (L*S2)*kds2*(-1*18*eta - 1*31*np.sqrt(-1*4*eta + 1) + 21))/((8*L*c**5))*PN5 + 1*((L*S1)*kds1*(-eta + 2*np.sqrt(-1*4*eta + 1) + 2) + (L*S2)*kds2*(-eta - 2*np.sqrt(-1*4*eta + 1) + 2))/(2*L*c**3)*PN3 + 1*(8*L*E*eta**2 - 1*22*E*eta*L + 4*L*E)/c**4*PN4 + 1*(2*L*eta - 1*4*L)/c**2*PN2
    d4 =  1*((L*S1)*kds1*(-1*18*eta**2 + 23*eta*np.sqrt(-1*4*eta + 1) + 21*eta - 1*24*np.sqrt(-1*4*eta + 1) - 24) + (L*S2)*kds2*(-1*18*eta**2 - 1*23*np.sqrt(-1*4*eta + 1)*eta + 21*eta + 24*np.sqrt(-1*4*eta + 1) - 24))/(8*L*c**5)*PN5 + 1*(5*L*eta**2 - 1*11*eta*L + (17*L)/2)/c**4*PN4
    d5 =  1*((L*S1)*kds1*(-1*15*np.sqrt(-1*4*eta + 1)*eta*L/4 + L*np.sqrt(-1*4*eta + 1) + L - 1*17*L*eta/4 + (9*L*eta**2)/4) + (L*S2)*kds2*(15*np.sqrt(-1*4*eta + 1)*eta*L/4 - L*np.sqrt(-1*4*eta + 1) + L - 1*17*L*eta/4 + (9*L*eta**2)/4))/c**5*PN5 + 1*(-L**3*eta**2 + (L**3*eta)/2)/c**4*PN4

    return n, np.sqrt(et2), ar, np.sqrt(er2), np.sqrt(ephi2), d2, d3, d4, d5, f_4t, f_5t, g_4t, g_5t


def dy_dt_2_5PN(y, t, t0, eta, S1, S2, t_eval, E_list, L_list, u_list, dk_list, dphi_list, radiation_reaction=False, spinning=True, PN=5) :

    PN2, PN3, PN4, PN5 = PN_param(PN)

    if spinning :

        n, et, kx, ky, kz, s1x, s1y, s1z, s2x, s2y, s2z, phi = y

        k = np.array([kx, ky, kz])/(kx**2 + ky**2 + kz**2)**0.5
        if S1 != 0 and S2 != 0 :
            s1 = np.array([s1x, s1y, s1z])/(s1x**2 + s1y**2 + s1z**2)**0.5
            s2 = np.array([s2x, s2y, s2z])/(s2x**2 + s2y**2 + s2z**2)**0.5
        else :
            s1 = np.array([0, 0, 0])
            s2 = np.array([0, 0, 0])

        kds1 = np.dot(k, s1)
        kds2 = np.dot(k, s2)

        E, L, ar, er, ephi, d2, d3, d4, d5, f_4t, f_5t, g_4t, g_5t = spinning_orbit_2_5PN_param(n, et, kds1, kds2, eta, S1, S2, t, PN=PN, analytic=analytic_E_L)

        t_eval.append(t)
        E_list.append(E)
        L_list.append(L)

        # solve Kepler's equation

        u1PN = mikkola(et, n*(t-t0))
        nu1PN = 2*np.arctan(np.sqrt((ephi + 1)/(ephi - 1))*np.tanh(u1PN/2))
        u = mikkola(et, n*(t-t0) - (f_4t + f_5t)*nu1PN - (g_4t + g_5t)*np.sin(nu1PN))

        u_list.append(u)

        dy = np.zeros(len(y))

        # dn/dt and det/dt

        beta = et*np.cosh(u) - 1

        if radiation_reaction :
            dy[0] = -n**(11/3)*8*eta/(5*beta**7) * (-49*beta**2 - 32*beta**3 + 35*(et**2-1)*beta - 6*beta**4 + 9*et**2*beta**2)
            dy[1] = -n**(8/3)*8*eta*(et**2-1)/(15*beta**7*et) * (-49*beta**2 - 17*beta**3 + 35*(et**2-1)*beta - 3*beta**4 + 9*et**2*beta**2)
        else :
            dy[0] = 0
            dy[1] = 0

        # precession equation

        r = ar*(er*np.cosh(u) - 1)

        s1crossk = np.cross(s1, k)
        s2crossk = np.cross(s2, k)

        f_3L =  PN3*(-1*eta/2 + np.sqrt(-1*4*eta + 1) + 1)*1/r**3
        f_5L =  PN5*(6*L**2*eta*(eta - np.sqrt(-1*4*eta + 1) - 1) + E*eta*r**2*(-1*18*eta + 31*np.sqrt(-1*4*eta + 1) + 21) + r*(-1*18*eta**2 + 23*eta*np.sqrt(-1*4*eta + 1) + 21*eta - 1*24*np.sqrt(-1*4*eta + 1) - 24))*1/(8*r**5)
        g_3L =  PN3*(-1*eta/2 - np.sqrt(-1*4*eta + 1) + 1)*1/r**3
        g_5L =  PN5*(6*L**2*eta*(eta + np.sqrt(-1*4*eta + 1) - 1) + E*eta*r**2*(-1*18*eta - 1*31*np.sqrt(-1*4*eta + 1) + 21) + r*(-1*18*eta**2 - 1*23*eta*np.sqrt(-1*4*eta + 1) + 21*eta + 24*np.sqrt(-1*4*eta + 1) - 24))*1/(8*r**5)

        dk = (f_3L + f_5L)*S1*s1crossk + (g_3L + g_5L)*S2*s2crossk
        dy[2], dy[3], dy[4] = dk[0], dk[1], dk[2]
        ds1 = -(f_3L + f_5L)*L*s1crossk
        dy[5], dy[6], dy[7] = ds1[0], ds1[1], ds1[2]
        ds2 = -(g_3L + g_5L)*L*s2crossk
        dy[8], dy[9], dy[10] = ds2[0], ds2[1], ds2[2]

        dk_list.append(dk)

        # dphi/dt

        dky = dy[3]
        dkx = dy[2]
        dalpha = (kx*dky - dkx*ky)/(kx**2 + ky**2)

        dy[11] = d2/r**2 + d3/r**3 + d4/r**4 + d5/r**5 - dalpha*kz
        dphi_list.append(dy[11])

        return dy
    
    else : 

        n, et, phi0 =  y

        kds1 = 0
        kds2 = 0

        E, L, ar, er, ephi, d2, d3, d4, d5, f_4t, f_5t, g_4t, g_5t = spinning_orbit_2_5PN_param(n, et, kds1, kds2, eta, S1, S2, t, PN=PN)

        t_eval.append(t)
        E_list.append(E)
        L_list.append(L)

        # solve Kepler's equation

        u1PN = mikkola(et, n*(t-t0))
        nu1PN = 2*np.arctan(np.sqrt((ephi + 1)/(ephi - 1))*np.tanh(u1PN/2))
        u = mikkola(et, n*(t-t0) - (f_4t + f_5t)*nu1PN - (g_4t + g_5t)*np.sin(nu1PN))

        u_list.append(u)

        dy = np.zeros(len(y))

        beta = et*np.cosh(u) - 1

        if radiation_reaction :
            dy[0] = -n**(11/3)*8*eta/(5*beta**7) * (-49*beta**2 - 32*beta**3 + 35*(et**2-1)*beta - 6*beta**4 + 9*et**2*beta**2)
            dy[1] = -n**(8/3)*8*eta*(et**2-1)/(15*beta**7*et) * (-49*beta**2 - 17*beta**3 + 35*(et**2-1)*beta - 3*beta**4 + 9*et**2*beta**2)
        else :
            dy[0] = 0
            dy[1] = 0

        return dy


def spinning_orbit_2_5PN(t, t0, eta, S1, S2, y0, PN=5, analytic_E_L=True, radiation_reaction=False, spinning=True, verbose=True, num_checks=False) :

    PN2, PN3, PN4, PN5 = PN_param(PN)

    if verbose : print('Computing orbit at ' + str(PN/2) + 'PN ========================\n')

    # find xi in terms of et0 and b

    b = y0[0]
    et0 = y0[1]
    phi0 = y0[-1]

    n0 = (np.sqrt(et0**2 - 1)/(b + np.sqrt(et0**2 - 1) * ((eta - 1)/(eta**2 - 1) + (7*eta - 6)/6)))**(3/2)
    yini = np.copy(y0)
    yini[0] = n0

    # solve differential system =================

    if verbose : print('Solving differential system...')

    t_eval, E_list, L_list, u_list, dk_list, dphi_list = [], [], [], [], [], []

    sol = odeint(dy_dt_2_5PN, yini, t, args=(t0, eta, S1, S2, t_eval, E_list, L_list, u_list, dk_list, dphi_list, radiation_reaction, spinning, PN, analytic_E_L))


    if spinning : 
        n, et, k, s1, s2, phi = sol[:,0], sol[:,1], sol[:,2:5], sol[:,5:8], sol[:,8:11], sol[:,11]

        # set phi(t0) = phi0
        phi_at_t0 = np.interp(t0, t, phi)
        phi -= phi_at_t0  + phi0 

        k = np.transpose(k)
        s1 = np.transpose(s1)
        s2 = np.transpose(s2)

        kds1 = dot(k, s1)
        kds2 = dot(k, s2)

    else :
        n, et = sol[:,0], sol[:,1]
        s1 = np.zeros((3, len(t)))
        s2 = np.zeros((3, len(t)))
        
        kds1 = 0
        kds2 = 0

    # get system parameters ===================

    if verbose : print('Getting system parameters...')

    E = np.interp(t, t_eval, E_list)
    L = np.interp(t, t_eval, L_list)

    n_2, et_2, ar, er, ephi, d2, d3, d4, d5, f_4t, f_5t, g_4t, g_5t = spinning_orbit_2_5PN_param_from_E_L(E, L, kds1, kds2, eta, S1, S2, PN=PN)


    # solve Kepler's equation (interpolating the u's already computed in odeint calls does not really work somehow...) =================
    
    u1PN = mikkola(et, n*(t-t0))
    nu1PN = 2*np.arctan(np.sqrt((ephi + 1)/(ephi - 1))*np.tanh(u1PN/2))
    u = mikkola(et, n*(t-t0) - (f_4t + f_5t)*nu1PN - (g_4t + g_5t)*np.sin(nu1PN))



    nu = 2*np.arctan(np.sqrt((ephi + 1)/(ephi - 1))*np.tanh(u/2))

    if verbose : print('Find motion of the orbital basis...')

    # get radial motion ===================

    r = ar*(er*np.cosh(u) - 1)

    # get orbital basis ===============

    if spinning : 

        k /= np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)

        iota = np.arccos(k[2])
        alpha = -np.arctan(k[0]/k[1])
        
        n_vec = np.array([np.cos(alpha)*np.cos(phi) - np.cos(iota)*np.sin(alpha)*np.sin(phi), np.sin(alpha)*np.cos(phi) + np.cos(iota)*np.cos(alpha)*np.sin(phi), np.sin(iota)*np.sin(phi)])
        xi_vec = cross(k, n_vec)

    else : 

        K = 1 + PN2*3/L**2 - 0.25*PN4*3*(-35 - 10*E*L**2 + 10*eta + 4*E*L**2*eta)/L**4
        f_4phi = -PN4*(1 + 2*E*L**2)*eta*(3*eta - 1)/(8*L**2)
        g_4phi = -PN4*3*(1 + 2*E*L**2*eta)**(3/2)*eta**2/(32*L**4)

        phi = phi0 + K*(nu + f_4phi*np.sin(2*nu) + g_4phi*np.sin(3*nu))

        n_vec = np.array([np.cos(phi), np.sin(phi), np.zeros(len(t))])
        k = np.array([np.zeros(len(t)), np.zeros(len(t)), np.ones(len(t))])
        xi_vec = cross(k, n_vec)


    # analytical derivatives =======================

    if verbose : print('Computing derivatives...')

    # radial derivative

    dnu_du = np.sqrt((ephi + 1)/(ephi - 1))/(np.cosh(u/2)**2 + (ephi + 1)/(ephi - 1)*np.sinh(u/2)**2)
    dt_du = (et*np.cosh(u) - 1 + (f_4t + f_5t + np.cos(nu)*(g_4t + g_5t))*dnu_du)/n
    dr = ar*er*np.sinh(u)/dt_du


    # position derivative

    if spinning : 
        dk_list = np.array(dk_list)
        dkx, dky, dkz, dphi = np.interp(t, t_eval, dk_list[:,0]), np.interp(t, t_eval, dk_list[:,1]), np.interp(t, t_eval, dk_list[:,2]), np.interp(t, t_eval, dphi_list)

        dk = np.array([dkx, dky, dkz])

        dalpha = (k[0]*dk[1] - dk[0]*k[1])/(k[0]**2 + k[1]**2)   
        diota = -dk[2]/np.sqrt(1-k[2]**2)

        dn_vec = (np.cos(iota)*dalpha + dphi)*xi_vec + (np.sin(phi)*diota - np.cos(phi)*np.sin(iota)*dalpha)*k

        v = dr*n_vec + r*dn_vec

    else :

        dphi = K*(1 + 2*f_4phi*np.cos(2*nu) + 3*g_4phi*np.cos(3*nu))*dnu_du/dt_du

        v = dr*np.array([np.cos(phi), np.sin(phi), np.zeros(len(t))]) + r*dphi*np.array([-np.sin(phi), np.cos(phi), np.zeros(len(t))])


    if verbose : print('Done !\n')

    # debug plots ===========================

    if num_checks :

        # deviation of orbital parameters from initial value

        create_plot(r'$t$ $(GM/c^3)$', r'$\left|\frac{\mathrm{param}-\mathrm{param_0}}{\mathrm{param_0}}\right|$', [t[0], t[-1]], title=str(PN/2)+'PN', logy=False)

        plt.plot(t, np.abs(et_2 - et_2[0])/np.abs(et_2[0]), label=r'$e_t$')
        plt.plot(t, np.abs(er - er[0])/np.abs(er[0]), label=r'$e_r$')
        plt.plot(t, np.abs(ephi - ephi[0])/np.abs(ephi[0]), label=r'$e_\varphi$')
        plt.plot(t, np.abs(ar - ar[0])/np.abs(ar[0]), label=r'$a_r$')
        plt.plot(t, np.abs(n - n[0])/np.abs(n[0]), label=r'$n$')

        plt.legend()

        # order of magnitude of the 2.5PN correction

        if PN == 5 :

            n1_5PN, et1_5PN, ar1_5PN, er1_5PN, ephi1_5PN, d2_1_5PN, d3_1_5PN, d4_1_5PN, d5_1_5PN, f_4t_1_5PN, f_5t_1_5PN, g_4t_1_5PN, g_5t_1_5PN = spinning_orbit_2_5PN_param_from_E_L(E, L, kds1, kds2, eta, S1, S2, PN=3)

            create_plot(r'$t$ $(GM/c^3)$', r'$\left|\frac{\mathrm{param}^{2.5}-\mathrm{param}^{1.5}}{\mathrm{param}^{1.5}}\right|$', [t[0], t[-1]], title=str(PN/2)+'PN', logy=True)
            
            plt.plot(t, np.abs(et_2 - et1_5PN)/np.abs(et1_5PN), label=r'$e_t$')
            plt.plot(t, np.abs(er - er1_5PN)/np.abs(er1_5PN), label=r'$e_r$')
            plt.plot(t, np.abs(ephi - ephi1_5PN)/np.abs(ephi1_5PN), label=r'$e_\varphi$')
            plt.plot(t, np.abs(ar - ar1_5PN)/np.abs(ar1_5PN), label=r'$a_r$')
            plt.plot(t, np.abs(n - n1_5PN)/np.abs(n1_5PN), label=r'$n$')

            plt.legend()

    return r, phi, n_vec, k, xi_vec, s1, s2, dr, v


def ADM2harmonic(r, dr, n, v, s1, s2, S1, S2, eta, PN=5) : 

    #print('PN = ', PN)

    PN2, PN3, PN4, PN5 = PN_param(PN)
    S = S1*s1 + S2*s2

    r_harm = r - PN3*0.5*eta*dot(S, cross(n, v)) + PN4*(eta*(5*dot(v, v) - 19*dr**2)/8 + (3*eta + 0.25)/r)
    n_harm = n + PN3*0.5*eta/r*(n*dot(S, cross(n, v)) - cross(S, v)) + PN4*0.25*9*dr*eta/r*(dr*n - v)
    v_harm = v - PN3*0.5*eta*cross(S, n)/r**2 + PN4*(eta*(dr*n*(3*dr**2 - 7*dot(v, v)) + v*(17*dr**2 - 13*dot(v, v)))/(8*r) + 0.25*((21*eta + 1)*v - (19*eta + 2)*dr*n)/r**2)

    dr_harm = dot(v_harm, n_harm)

    return r_harm, dr_harm, n_harm, v_harm
    


# Gravitational waves emission ========================================================================================

def GW_emission_from_orbit(Theta, R, t, n, velocity, r, dr, s1, s2, m1, m2, chi1, chi2, GW_order = 4) :

    m = m1 + m2
    eta = m1*m2/m**2

    # New sky basis in the x,y,z basis

    N = np.array([np.sin(Theta)*np.ones(len(t)), np.zeros(len(t)), np.cos(Theta)*np.ones(len(t))])
    p = np.array([np.zeros(len(t)), -np.ones(len(t)), np.zeros(len(t))])
    q = np.array([np.cos(Theta)*np.ones(len(t)), np.zeros(len(t)), -np.sin(Theta)*np.ones(len(t))])

    # Compute dot products
        
    pdn = dot(p, n)
    qdn = dot(q, n)
    Ndn = dot(N, n)
    
    pddr = dot(p, velocity)
    qddr = dot(q, velocity)
    Nddr = dot(N, velocity)

    # Compute cross products        
    
    pds1cN = dot(p, cross(s1, N))
    pds2cN = dot(p, cross(s2, N))
    qds1cN = dot(q, cross(s1, N))
    qds2cN = dot(q, cross(s2, N))


    # Compute polarisations

    PN1, PN2, PN3, PN4 = [1 if i<=GW_order else 0 for i in range(4)]

    z = 1/r
    v = dot(velocity, velocity)**(1/2)

    delta = np.abs(m1-m2)/m
    X1 = m1/m
    X2 = m2/m

    # h_plus
    LO_p = PN1*(qdn**2 - pdn**2)*z + pddr**2 - qddr**2
    NLO_p = - PN2*0.5*delta * ((Ndn * dr - Nddr)*z*pdn**2 - 6*z*Ndn*pdn*pddr + (-3*Ndn*dr + Nddr)*z*qdn**2 + 6*z*Ndn*qdn*qddr + 2 * (pddr**2 - qddr**2)*Nddr)
    NNLO_p = PN3*1/6 * (6*Nddr**2*(pddr**2-qddr**2)*(1-3*eta)
    + ((6*eta-2)*Nddr**2*pdn**2 + (96*eta-32)*Nddr*Ndn*pddr*pdn + (-6*eta+2)*Nddr**2*qdn**2 + (-96*eta+32)*Nddr*Ndn*qddr*qdn + ((-14+42*eta)*Ndn**2 - 4 + 6*eta)*pddr**2 + ((-42*eta+14)*Ndn**2 + 4 - 6*eta)*qddr**2)*z
    + ((-9*eta+3)*pddr**2 + (-3+9*eta)*qddr**2)*v**2
    + ((29 + (7-21*eta)*Ndn**2)*pdn**2 + (-29 + (21*eta-7)*Ndn**2)*qdn**2)*z**2
    + (((-9*eta+3)*Ndn**2 - 10 - 3*eta)*pdn**2 + ((-3+9*eta)*Ndn**2 + 10 + 3*eta)*qdn**2)*z*v**2
    + ((-36*eta + 12)*Nddr*Ndn*pdn**2 + ((-90*eta+30)*Ndn**2 + 12*eta + 20)*pddr*pdn + (-12+36*eta)*Nddr*Ndn*qdn**2 + ((90*eta-30)*Ndn**2 - 12*eta - 20)*qddr*qdn)*z*dr
    + (((45*eta-15)*Ndn**2 - 9*eta + 3)*pdn**2 + ((15-45*eta)*Ndn**2 - 3 + 9*eta)*qdn**2)*z*dr**2)
    NNLO_SO_p = PN4*z**2*(pdn*(X2*chi2*pds2cN - X1*chi1*pds1cN) + qdn*(X1*chi1*qds1cN - X2*chi2*qds2cN))

    # h_cross
    LO_c = -PN1*pdn*qdn*z + pddr*qddr
    NLO_c = -PN2*delta*((((3*Ndn*dr - Nddr)*qdn - 3*Ndn*qddr)*pdn - 3*Ndn*qdn*pddr)*z + 2*pddr*qddr*Nddr)
    NNLO_c = PN3*1/6 * (6*(1-3*eta)*Nddr**2*pddr*qddr 
    + (((6*eta-2)*Nddr**2*qdn + (48*eta-16)*Nddr*Ndn*qddr)*pdn + (48*eta-16)*Nddr*Ndn*pddr*qdn + ((-14+42*eta)*Ndn**2 - 4 + 6*eta)*qddr*pddr)*z
    + (-9*eta+3)*qddr*pddr*v**2
    + (29 + (7-21*eta)*Ndn**2)*qdn*pdn*z**2
    + ((-9*eta+3)*Ndn**2 - 10 - 3*eta)*qdn*pdn*z*v**2
    + (((-36*eta+12)*Nddr*Ndn*qdn + ((15-45*eta)*Ndn**2 + 10 + 6*eta)*qddr)*pdn + ((15-45*eta)*Ndn**2 + 10 + 6*eta)*pddr*qdn)*dr*z
    + ((45*eta-15)*Ndn**2 - 9*eta + 3)*qdn*pdn*dr**2*z)
    NNLO_SO_c = PN4*z**2*qdn*(X2*chi2*pds2cN - X1*chi1*pds1cN)

    h_plus  = LO_p + NLO_p + NNLO_p + NNLO_SO_p
    h_cross = LO_c + NLO_c + NNLO_c + NNLO_SO_c

    return 2*h_plus, 4*h_cross