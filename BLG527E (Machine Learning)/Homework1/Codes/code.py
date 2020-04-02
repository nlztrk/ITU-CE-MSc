import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


xma=30.0
xmi=-20.0
xint=800
gap = (xma-xmi)/xint
xs = np.linspace(xmi, xma, num=xint) ## Defining the x axis

####
#### FUNCTIONS ###########################
####

mu1 = -5 ## Example decision surface finder function
mu2 = 10 ## Example decision surface finder function
ssd1 = math.sqrt(4**2) ## Example decision surface finder function
ssd2 = math.sqrt(2*4**2) ## Example decision surface finder function
prob1 = 0.5
prob2 = 0.5

def decsurf_eq(x): ## Example decision surface finder function

	ePart1 = math.pow(np.e, -(x-mu1)**2/(2*ssd1**2))
	eq1 = ((1.0 / (math.sqrt(2*math.pi)*ssd1)) * ePart1)*prob1

	ePart2 = math.pow(math.e, -(x-mu2)**2/(2*ssd2**2))
	eq2 = ((1.0 / (math.sqrt(2*math.pi)*ssd2)) * ePart2)*prob2

	return eq2-eq1

x0 = 1 ## Example decision surface finder function
surfaces = fsolve(decsurf_eq,x0) ## Example decision surface finder function
print("Function output example: The 1.a. seperator is on x("+str(surfaces[0])+")") ## Example decision surface finder function

def pdf(mean, ssd, x, prob): ## Defining the PDF function

 ePart = np.power(np.e, -(x-mean)**2/(2*ssd**2)) ## exp part of the formula

 return ((1.0 / (np.sqrt(2*np.pi)*ssd)) * ePart)*prob ## returns the normal dist value for the given x

def decision_surface(g1, g2): ## Defining the decision surface function
	index=0
	for step in xs:
		
		if (g1[index]<g2[index]):
			return xs[index] ## Returns the separation point x
		index=index+1

def max_event(g1, g2, g3): ## Defining the select maximum discrete function
    index=0
    max_array=[]
    for step in xs:

	    if (g1[index] >= g2[index]) and (g1[index] >= g3[index]): 
	        largest = 0 
	    elif (g2[index] >= g1[index]) and (g2[index] >= g3[index]): 
	        largest = 1
	    else: 
	        largest = 2
	    max_array.append(largest)
	    index = index + 1

    return max_array ## Returns an array valued as class index number for each x

####
#### FUNCTIONS ###########################
####

print()

####
#### 1.A ###########################
####

#Defining question parameters

m = [-5,10]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2)]
prob = [0.5,0.5]

g=[0,0] #Discrete function outputs



pdf1 = pdf(m[0], ssd[0], xs, prob[0]) ## Defining PDFs
pdf2 = pdf(m[1], ssd[1], xs, prob[1]) ## Defining PDFs

g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs

mu1 = m[0]
mu2 = m[1]
ssd1 = ssd[0]
ssd2 = ssd[1]
prob1 = prob[0]
prob2 = prob[1]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.A Decision Surface at "+str(surfaces[0]))

plt.plot(xs, pdf1, label='PDF of x : 1')
plt.plot(xs, pdf2, label='PDF of x : 2')
plt.axvline(x=surfaces[0], color="black", label='Seperating Surface (x='+str(round(surfaces[0],2))+')')
plt.legend()
plt.show()



####
#### /1.A ###########################
####

print()

####
#### 1.B ###########################
####

#Defining question parameters

m = [-5,10,5]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2),math.sqrt(3*4**2)]
prob = [0.1,0.9]

g=[0,0] #Discrete function outputs

pdf1 = pdf(m[0], ssd[0], xs, prob[0]) ## Defining PDFs
pdf2 = pdf(m[1], ssd[1], xs, prob[1]) ## Defining PDFs

g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs

mu1 = m[0]
mu2 = m[1]
ssd1 = ssd[0]
ssd2 = ssd[1]
prob1 = prob[0]
prob2 = prob[1]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.B Decision Surface at "+str(surfaces[0]))

plt.plot(xs, pdf1, label='PDF of x : 1')
plt.plot(xs, pdf2, label='PDF of x : 2')
plt.axvline(x=surfaces[0], color="black", label='Seperating Surface (x='+str(round(surfaces[0],2))+')')
plt.legend()
plt.show()

####
#### /1.B ###########################
####

print()

####
#### 1.C ###########################
####

## Ns=10 for (a)

#Defining question parameters

m = [-5,10]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2)]

probs1 = [0.5,0.5]
probs2 = [0.1,0.9]

g=[0,0] #Discrete function outputs

samples1 = np.random.normal(m[0], ssd[0], 10)
samples2 = np.random.normal(m[1], ssd[1], 10)

newmeans = [np.mean(samples1), np.mean(samples2)]
newstds = [np.std(samples1), np.std(samples2)]

pdf1 = pdf(newmeans[0], ssd[0], xs, probs1[0]) ## Defining PDFs
pdf2 = pdf(newmeans[1], ssd[1], xs, probs1[1]) ## Defining PDFs
g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs

plt.hist(samples1, 10, density=True) # Creating histograms
plt.hist(samples2, 10, density=True) # Creating histograms

mu1 = newmeans[0]
mu2 = newmeans[1]
ssd1 = newstds[0]
ssd2 = newstds[1]
prob1 = probs1[0]
prob2 = probs1[1]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.C.1 Decision Surface at "+str(surfaces[0]))

plt.axvline(x=surfaces[0], color="black", label='Seperating Surface (x='+str(round(surfaces[0],2))+')')

plt.legend()
plt.show()
print("1.C.1 New Means mu1: "+str(round(newmeans[0],2))+", mu2: "+str(round(newmeans[1],2)))
print("1.C.1 New Std Devs sigma1: "+str(round(newstds[0],2))+", sigma2: "+str(round(newstds[1],2)))
## /Ns=10 for (a)

print()

## Ns=100 for (a)

#Defining question parameters

m = [-5,10]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2)]

probs1 = [0.5,0.5]
probs2 = [0.1,0.9]

g=[0,0] #Discrete function outputs

samples1 = np.random.normal(m[0], ssd[0], 100)
samples2 = np.random.normal(m[1], ssd[1], 100)

newmeans = [np.mean(samples1), np.mean(samples2)]
newstds = [np.std(samples1), np.std(samples2)]

pdf1 = pdf(newmeans[0], ssd[0], xs, probs1[0]) ## Defining PDFs
pdf2 = pdf(newmeans[1], ssd[1], xs, probs1[1]) ## Defining PDFs
g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs

plt.hist(samples1, 100, density=False) # Creating histograms
plt.hist(samples2, 100, density=False) # Creating histograms

mu1 = newmeans[0]
mu2 = newmeans[1]
ssd1 = newstds[0]
ssd2 = newstds[1]
prob1 = probs1[0]
prob2 = probs1[1]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.C.2 Decision Surface at "+str(surfaces[0]))

plt.axvline(x=surfaces[0], color="black", label='Seperating Surface (x='+str(round(surfaces[0],2))+')')

plt.legend()
plt.show()
print("1.C.2 New Means mu1: "+str(round(newmeans[0],2))+", mu2: "+str(round(newmeans[1],2)))
print("1.C.2 New Std Devs sigma1: "+str(round(newstds[0],2))+", sigma2: "+str(round(newstds[1],2)))
## /Ns=100 for (a)

print()

## Ns=10 for (b)

#Defining question parameters

m = [-5,10]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2)]

probs1 = [0.5,0.5]
probs2 = [0.1,0.9]

g=[0,0] #Discrete function outputs

samples1 = np.random.normal(m[0], ssd[0], 10)
samples2 = np.random.normal(m[1], ssd[1], 10)

newmeans = [np.mean(samples1), np.mean(samples2)]
newstds = [np.std(samples1), np.std(samples2)]

pdf1 = pdf(newmeans[0], ssd[0], xs, probs2[0]) ## Defining PDFs
pdf2 = pdf(newmeans[1], ssd[1], xs, probs2[1]) ## Defining PDFs
g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs

plt.hist(samples1, 10, density=True) # Creating histograms
plt.hist(samples2, 10, density=True) # Creating histograms

mu1 = newmeans[0]
mu2 = newmeans[1]
ssd1 = newstds[0]
ssd2 = newstds[1]
prob1 = probs2[0]
prob2 = probs2[1]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.C.3 Decision Surface at "+str(surfaces[0]))

plt.axvline(x=surfaces[0], color="black", label='Seperating Surface (x='+str(round(surfaces[0],2))+')')

plt.legend()
plt.show()
print("1.C.3 New Means mu1: "+str(round(newmeans[0],2))+", mu2: "+str(round(newmeans[1],2)))
print("1.C.3 New Std Devs sigma1: "+str(round(newstds[0],2))+", sigma2: "+str(round(newstds[1],2)))
## /Ns=10 for (b)

print()

## Ns=100 for (b)

#Defining question parameters

m = [-5,10]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2)]

probs1 = [0.5,0.5]
probs2 = [0.1,0.9]

g=[0,0] #Discrete function outputs

samples1 = np.random.normal(m[0], ssd[0], 100)
samples2 = np.random.normal(m[1], ssd[1], 100)

newmeans = [np.mean(samples1), np.mean(samples2)]
newstds = [np.std(samples1), np.std(samples2)]

pdf1 = pdf(newmeans[0], ssd[0], xs, probs2[0]) ## Defining PDFs
pdf2 = pdf(newmeans[1], ssd[1], xs, probs2[1]) ## Defining PDFs
g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs

plt.hist(samples1, 100, density=True) # Creating histograms
plt.hist(samples2, 100, density=True) # Creating histograms

mu1 = newmeans[0]
mu2 = newmeans[1]
ssd1 = newstds[0]
ssd2 = newstds[1]
prob1 = probs2[0]
prob2 = probs2[1]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.C.4 Decision Surface at "+str(surfaces[0]))

plt.axvline(x=surfaces[0], color="black", label='Seperating Surface (x='+str(round(surfaces[0],2))+')')

plt.legend()
plt.show()
print("1.C.4 New Means mu1: "+str(round(newmeans[0],2))+", mu2: "+str(round(newmeans[1],2)))
print("1.C.4 New Std Devs sigma1: "+str(round(newstds[0],2))+", sigma2: "+str(round(newstds[1],2)))
## /Ns=100 for (b)

####
#### /1.C ###########################
####

print()

####
#### 1.D ###########################
####

#Defining question parameters

m = [-5,10,5]
ssd = [math.sqrt(4**2),math.sqrt(2*4**2),math.sqrt(3*4**2)]
prob = [0.4,0.4,0.2]

g=[0,0,0] #Discrete function outputs

pdf1 = pdf(m[0], ssd[0], xs, prob[0])
pdf2 = pdf(m[1], ssd[1], xs, prob[1])
pdf3 = pdf(m[2], ssd[2], xs, prob[2])

g[0]=np.log(pdf1) # Defining discrete outputs
g[1]=np.log(pdf2) # Defining discrete outputs
g[2]=np.log(pdf3) # Defining discrete outputs

maxes=max_event(g[0],g[1],g[2]) ## Getting max discrete values

mu1 = m[0]
mu2 = m[2]
ssd1 = ssd[0]
ssd2 = ssd[2]
prob1 = prob[0]
prob2 = prob[2]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.D First Decision Surface at "+str(surfaces[0]))

mu1 = m[1]
mu2 = m[2]
ssd1 = ssd[1]
ssd2 = ssd[2]
prob1 = prob[1]
prob2 = prob[2]
x0 = 1
surfaces = fsolve(decsurf_eq,x0) # Solving the decision surface equation
print("1.D Second Decision Surface at "+str(surfaces[0]))

plt.plot(xs, pdf1, label='PDF of x : 1')
plt.plot(xs, pdf2, label='PDF of x : 2')
plt.plot(xs, pdf3, label='PDF of x : 3')

plt.plot(0, 0, label='x:1 Decision Region' , color='paleturquoise')
plt.plot(0, 0, label='x:2 Decision Region' , color='peachpuff')
plt.plot(0, 0, label='x:3 Decision Region' , color='palegreen')

plotindex=0

for element in maxes: ## Defining the decision regions by printing each interval
	if element==0:
		plt.axvspan(xs[plotindex], xs[plotindex]+gap, alpha=1, color='paleturquoise')
	if element==1:
		plt.axvspan(xs[plotindex], xs[plotindex]+gap, alpha=1, color='peachpuff')		
	if element==2:
		plt.axvspan(xs[plotindex], xs[plotindex]+gap, alpha=1, color='palegreen')

	plotindex = plotindex + 1

plt.legend()
plt.show()

####
#### /1.D ###########################
####

print()