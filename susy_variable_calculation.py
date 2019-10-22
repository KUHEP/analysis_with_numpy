import numpy as np

### Calculating M_lb
### need pt, eta, phi, m of bjet and lep

px_bjet = np.abs(pt1_goodevbjet)*np.cos(phi1_goodevbjet)
py_bjet = np.abs(pt1_goodevbjet)*np.sin(phi1_goodevbjet)
pz_bjet = np.abs(pt1_goodevbjet)*np.sinh(eta1_goodevbjet)
p_vec_bjet = np.array([px1_bjet, py1_bjet, pz1_bjet])

e_bjet = np.sqrt(np.pow(p_vec_bjet, 2) + np.pow(m1_goodevbjet, 2))

px_lep = np.abs(pt1_goodevtlep)*np.cos(phi1_goodevtlep)
py_lep = np.abs(pt1_goodevtlep)*np.sin(phi1_goodevtlep)
pz_lep = np.abs(pt1_goodevtlep)*np.sinh(eta1_goodevtlep)
p_vec_lep = np.array([px_lep, py_lep, pz_lep])

e_lep = np.sqrt(np.pow(p_vec_lep, 2) + np.pow(m1_goodevtlep, 2))

p_vec_lb = p_vec_bjet + p_vec_lep
e_vec_lb = e_bjet + e_lep


mag_lb_sq = np.power(e_vec_lb, 2) - np.sum(np.power(p_vec_lb, 2), axis=0)

mag_lb = np.array([np.sqrt(mag) if mag > 0 else -np.sqrt(-mag) for mag in mag_lb_sq])



### Calculating MT
### need pt, phi of lep, MET, phi MET

dphi_lmet = phi_lep - phi_met

dphi_lmet = np.array([ phi + 2*np.pi if phi < -np.pi else phi for phi in dphi_lmet])
dphi_lmet = np.array([ phi - 2*np.pi if phi > np.pi else phi for phi in dphi_lmet])

mt = np.sqrt(2 * pt_lep * met * (1-cos(dphi_lmet)))



### Caltulcating min delta phi

### need phi of j1 and j2, phi of MET

dphi_j1met = phi_jet1 - phi_met
dphi_j1met = np.array([ phi + 2*np.pi if phi < -np.pi else phi for phi in dphi_j1met])
dphi_j1met = np.array([ phi - 2*np.pi if phi > np.pi else phi for phi in dphi_j1met])

dphi_j2met = phi_jet1 - phi_met
dphi_j2met = np.array([ phi + 2*np.pi if phi < -np.pi else phi for phi in dphi_j2met])
dphi_j2met = np.array([ phi - 2*np.pi if phi > np.pi else phi for phi in dphi_j2met])

min_dphi_j1j2met = np.min([dphi_j1met, dphi_j2met], axis=0)
