import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import corner

"""Function: Fit X ray, UV, and FWHM data for a set of quasars to a cosmological model, 
using Monte Carlo Markov Chain fits and cutting out some outliers using sigma clipping."""

readfile = 'Desktop/Arcetri1718/Data/sample_lia3.txt'
dat = open(readfile, 'r')
writefile = open('Desktop/Arcetri1718/sigmaclip_FWHM_log.txt', 'w')

z = []
logUV = []
logX = []
logfluxUV = []
logfluxX = []
errX = []
logFWHM = []

avg_sum = 0
err_sum = 0

for line in dat:
    line = line.split()
    if float(line[6])>1000:
        z.append(float(line[0]))
        logUV.append(float(line[1]))
        logX.append(float(line[2]))
        logfluxUV.append(float(line[3]))
        logfluxX.append(float(line[4]))
        errX.append(float(line[5]))
        logFWHM.append(np.log10(float(line[6])))
    
#print np.mean(logUV)
    
# define the function/model to fit
def mod1(xx, alpha, beta, gamma):
    (UV, FWHM) = xx
    return alpha*(UV+27.4289985587) + gamma*(FWHM-np.mean(logFWHM)) + beta
#+27.4289985587 for fluxes
#-30.258448414 for lumin
    
# define the likelihood function
# theta is the parameters
def lnlike(theta, UV, FWHM, y, yerr):
    alpha, beta, gamma, disp = theta
    snsquare = yerr**2+disp**2
    return -np.sum(((y-mod1((UV, FWHM), alpha, beta, gamma))**2/snsquare)+(np.log(snsquare)))
    
# then define the priors
def lnprior(theta):
    alpha, beta, gamma, disp = theta
    # chosen completely randomly (shouldn't matter for now)
    if -100 < alpha < 100 and -100 < beta < 100 and -100 < gamma < 100 and -100 < disp < 100:
        return 0.0
    return -np.inf
    
# then combine into full log-probability function
def lnprob(theta, xx, y, yerr):
    (UV,FWHM)=xx
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, UV, FWHM, y, yerr)
    
# divide into bins of log z = 0.08 and loop through, do the basic mcmc for each bin
# for each, if possible save the plot to the folder Test_bins_plots 
# and print out results to check that alpha is roughly the same for all 

#function for doing mcmc on a bin
def analyse_bin(indices, num, startpoint, endpoint):
    binUV = []
    binFWHM = []
    binX = []
    binerrX = []
    binz = []
    for i in indices:
        # create the data lists for just this bin 
        binUV.append(logfluxUV[i])
        binFWHM.append(logFWHM[i])
        binX.append(logfluxX[i])
        binerrX.append(errX[i])
        binz.append(z[i])
    binUV = np.array(binUV)
    binX = np.array(binX)
    binFWHM = np.array(binFWHM)
    binerrX = np.array(binerrX)
    binz = np.array(binz)
    
    writefile.write(str(10**startpoint)+' < z < '+str(10**endpoint)+': '+str(len(binUV))+'\n')

    # optimize normally, used for getting initial starting pos of walkers later  
    guess = (0.6, -31, 1)
    xx = (binUV, binFWHM)
    result, cov = op.curve_fit(mod1, xx, binX, guess, sigma=binerrX)
    # set up starting positions for walkers
    # use the starting params from the optimize for good estimate, then start in random places nearby
    ndim, nwalkers = 4, 100
    # also need to add guess of the dispersion
    guessdisp = 0.3
    pos = [np.array([result[0], result[1], result[2], guessdisp]) + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
    # set up an emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xx, binX, binerrX))
    # run mcmc
    sampler.run_mcmc(pos, 500)
    # 100 is the amount to ignore or 'cut off'
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    # obtain your final result/fit by 'mapping' out the samples
    alpha1, beta1, gamma1, disp1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))
    
    # sigma clipping
    cutoff=2.6
    # cuts is a list of indices of clipped points in the current sub list of this bin
    cuts = []
    new_indices, newbinUV, newbinFWHM, newbinX, newbinerrX, newbinz, cutUV, cutX, cuterrX = [], [], [], [], [], [], [], [], []
    for i in range(len(indices)):
        Erri = np.sqrt(disp1[0]**2+binerrX[i]**2)
        #Erri = binerrX[i]
        I = (binX[i] - mod1((binUV[i], binFWHM[i]), alpha1[0], beta1[0], gamma1[0]))/Erri
        if np.abs(I) >= cutoff:
            #print binUV[i], binX[i]
            cuts.append(i)
            cutUV.append(binUV[i])
            cutX.append(binX[i])
            cuterrX.append(binerrX[i])
    for i in range(len(indices)):
        if i not in cuts:
            new_indices.append(indices[i])
            newbinUV.append(binUV[i])
            newbinFWHM.append(binFWHM[i])
            newbinX.append(binX[i])
            newbinerrX.append(binerrX[i])
            newbinz.append(binz[i])
    
   # plots: data points
    plt.errorbar(newbinUV, newbinX, yerr=newbinerrX, fmt='o', markersize=2, color='k')
    # plot the points that were cut
    plt.errorbar(cutUV, cutX, yerr=cuterrX, fmt='o', markersize=4, color='r')
    plt.xlabel('log (L_UV)')
    plt.ylabel('log (L_X)')
    plt.ylim(-33, -29.5)
    plt.xlim(-29, -25)
    
    # make it recursive sigma clipping
    # base case (nothing left to be clipped, leave recursive function)
    if len(cuts) == 0:
        writefile.write('No more sigma clipping needed! Final results:'+'\n'+"Bin "+str(num)+': '+str(alpha1)+' '+str(beta1)+' '+str(gamma1)+' '+str(disp1)+'\n \n ----------------------------------------------------------------------- \n')
        plt.plot([-29, -25], [mod1((-29,3.594), alpha1[0], beta1[0], gamma1[0]), mod1((-25,3.594), alpha1[0], beta1[0], gamma1[0])], color='r', lw=2, label='MCMC with disp.')
        plt.text(-28.7,-29.7,r'z ~ '+str(round(np.mean(newbinz),3)),fontsize=12)
        plt.text(-28.7,-29.9,r'Data points (after $\sigma$-clip): '+str(len(newbinUV)), fontsize=12)
        plt.text(-28.7,-30.1,r'$\alpha$: '+str(round(alpha1[0],3))+r', $\beta$: '+str(round(beta1[0],3))+r', $\gamma$: '+str(round(gamma1[0],3)))
        plt.savefig('Desktop/Arcetri1718/Binplots_FWHM_sigclip/Bin'+str(binnum)+'.png')
        # add to bin points to plot v redshift
        zs.append(np.mean(newbinz))
        alphas.append(alpha1[0])
        betas.append(beta1[0])
        alpha_err.append(alpha1[1])
        beta_err.append(beta1[1])
        gammas.append(gamma1[0])
        gamma_err.append(gamma1[1])
        #update sums for computing average
        alphaxerr = (alpha1[0]/(alpha1[1])**2)
        err = (1/alpha1[1])**2
        return alphaxerr, err
    # not the base case (still needs clipping)
    else:
        writefile.write('Found '+str(len(cuts))+' points to clip.'+'\n'+"Bin "+str(num)+': '+str(alpha1[0])+', '+str(beta1[0])+', '+str(gamma1[0])+', '+str(disp1[0])+'\n \n')
        return analyse_bin(new_indices, num, startpoint, endpoint) 
    
startpoint = np.log10(0.25)
endpoint = startpoint
binnum = 0
zs = []
alphas = []
betas = []
alpha_err = []
beta_err = []
gammas = []
gamma_err = []

while endpoint < np.log10(3):
    endpoint+=0.1
    binnum+=1
    current_bin = []
    for i in range(len(z)):
        if startpoint <= np.log10(z[i]) < endpoint:
            #append all the indices of the points in this bin
            current_bin.append(i)
    #run mcmc on the current bin
    fig = plt.figure()
    alphaxerr, err = analyse_bin(current_bin, binnum, startpoint, endpoint)
    avg_sum += alphaxerr
    err_sum += err
    startpoint=endpoint
    
"""# plot results v redshift
plt.errorbar(zs, alphas, yerr=alpha_err, fmt='o', markersize=2, color='k')
plt.xlabel('z')
plt.ylabel(r'$\alpha$')
# with full, non-bin mcmc for comparison
#plt.plot([0,6], [0.617, 0.617], lw=3, color='r', alpha=0.5)
plt.show()"""

avg_alpha = avg_sum/err_sum
print avg_alpha

"""chisquaresum = 0
#reduced chi square
for i in range(len(alphas)):
    chisquaresum += (((alphas[i]-avg_alpha)/alpha_err[i])**2)
chisquare = chisquaresum/(len(alphas)-1)
print chisquare"""

# alpha is 0.588 with sigclip

dat.close()
writefile.close()

    
    
    
 

    
    
    