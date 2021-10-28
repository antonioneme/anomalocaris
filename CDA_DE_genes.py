import	argparse, sys, math, random
import numpy as np
import matplotlib.pyplot as plt

from pyod.models.lof import LOF
from pyod.models.iforest import IForest

from sklearn import cluster

from scipy.spatial.distance import euclidean

import	seaborn as sns


def	read_data(FF, n):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	G = {}
	for line in x:
		if line[0] != '#':
			totTrt = 0.0
			totCtr = 0.0
			xx = line.split('\t')
			Trt = []
			for i in range(1, n + 1):
				v = float(xx[i])
				totTrt = totTrt + v
				Trt.append(v)
			ln = len(xx)
			Ctr = []
			for i in range(n+1, ln):
				v = float(xx[i])
				totCtr = totCtr + v
				Ctr.append(v)
			if totCtr > 0:
				v = totTrt/totCtr
				#print "v = ", xx[0], v, Trt, totTrt, Ctr, totCtr
				if v > 0:
					fe = math.log(v,10.0)
				else:
					#fe = 0
					fe = -totCtr
					#cc = sys.stdin.read(1)
			else:
				fe = totTrt

			TrtX = []
			CtrX = []
			for jj,T in enumerate(Trt):
				if totTrt > 0:
					TrtX.append(T/totTrt)
				else:
					TrtX.append(0.0)
				if totCtr > 0:
					CtrX.append(Ctr[jj]/totCtr)
				else:
					CtrX.append(0.0)
			dTC = euclidean(TrtX, CtrX)
			G[xx[0]] = [fe, Trt, Ctr, totTrt, totCtr, dTC]
	return G

def	save_centroids(F_centroids, Cts, Num):
	f = open(F_centroids, "w")
	for i,c in enumerate(Cts):
		#print "i = ", i, c
		#print "Ni = ", Num[i]
		f.write(str(c) + "\t" + str(Num[i]) + "\n")
	f.close()

def	save_centroidsC(F_centroids, nBsqrd, Cts, Num):
	f = open(F_centroids, "w")
	for i in range(nBsqrd):
		for j in range(nBsqrd):
			f.write(str(i) + "\t" + str(j) + "\t" + str(Num[(i,j)]) + "\n")
	f.close()

def	position(a, L):
	for j,v in enumerate(L):
		if a == v[1]:
			return j
	return -1


def	is_in(a, L):
	for pp in L:
		if pp[0] == a[0] and pp[1] == a[1]:
			return 1
	return -1

def	opt_numBin(ErrnB, nBnum):
	ln = len(ErrnB)
	print "ln = ", ln
	# Remember, the min number of bins is 2
	dEs = abs(ErrnB[3] - ErrnB[2])
	for q in range(3, ln-1):
		dE = abs(ErrnB[q] - ErrnB[q+1])
		if dE > 0:
			R = dEs/dE
		else:
			R = 0
		print "q = ", q, R, ErrnB[q], ErrnB[q+1]
		#if R >= 0.05:
		if R >= 0.025:
			# Check if all centroids have at least a certain number of points
			cont = 1
		else:
			#if nBnum[ii][
			optnB = q
			#break
			return optnB
		dEs = dE
	return ln

"""
Each gene g is represented as {Trt(g), Ctr(g) }, where Trt(g) =  [repT1, repT2, ... repTn] and
Ctr(g) = [repC1, repC2, ...m repCn].
Gene g is then transformed to CDA coordinates in a simplex S. Let:
d1 = repT1 + repT2 + ... + repTn
d2 = repC1 + repC2 + ... + repCn

The position of gene g in the simplex S(Trt) is given by:

S(g,Trt) = {repT1/d1, repT2/d1, ..., repTn/d1}

and its position in the simplex S(Ctr) is given by:

S(g, Ctr) = {repC1/d2, repC2/d2, ..., repCn/d2}

The distance between the treated and control CDA (Compositional data Analysis) coordinates
is simple:

d = dist( S(g,Trt), S(g,Ctr) )

"""
def	create_simplex(nn, G, nB, msamp, F_centroids, FC_centroids, autom):

	F_FE_centroids = F_centroids + "_Transpose"

	# max distance in simplex
	mxdS = -1000000.0
	# min distance in simplex
	mndS = 10000000.0

	# max coefficient of dispersion
	mxCD = -10000000.0
	mnCD = 10000000.0

	# min and max FE
	mxFE = -1000000000.0
	mnFE = 1000000000.0

	# Map gene g to the simplex coordinates for CDA
	GeneSimplex = {}

	# The center of the composition
	cC = [1.0/float(nn)] * nn

	# Distances in the simplex
	L_Dist = []
	# G[g] = [fe, Trt, Ctr, totTrt, totCtr, dist(T,C)]

	# possible FEs
	L_FE_Dist = []

	# Dispersion for treatment
	L_Disp = []
	for g in G:
		d1 = 0.0
		for r in G[g][1]:
			d1 = d1 + r
		#print "g = ", g, d1
		d2 = 0.0
		for r in G[g][2]:
			d2 = d2 + r
		PosTrt = []
		if d1 > 0:
			for r in G[g][1]:
				l = r/d1
				PosTrt.append(l)
		else:
			PosTrt = [1.0/float(nn)] * nn
			#PosTrt = [0.0] * nn
		PosCtr = []
		if d2 > 0:
			for r in G[g][2]:
				l = r/d2
				PosCtr.append(l)
		else:
			PosCtr = [0.0] * nn

		# The distance between the treated and control cases for gene g
		# in the simplex for CDA
		dS = euclidean(PosTrt, PosCtr)
		if dS >= mxdS:
			mxdS = dS
		if dS <= mndS:
			mndS = dS

		L_Dist.append(tuple([dS]))

		L_FE_Dist.append(tuple([G[g][0]]))

		delta_T = euclidean(PosTrt, cC)
		delta_C = euclidean(PosCtr, cC)

		L_Disp.append(tuple([delta_T]))

		if delta_T < mnCD:
			mnCD = delta_T
		if delta_T > mxCD:
			mxCD = delta_T

		if G[g][0] > mxFE:
			mxFE = G[g][0]
		if G[g][0] < mnFE:
			mnFE = G[g][0]

		# GeneSimplex[g] = [dist in simplex, PosTrt, PosCtr, FE, totTreat, totCtr, delta_T, deltaa_C]
		GeneSimplex[g] = [ dS, PosTrt, PosCtr, G[g][0], G[g][3], G[g][4], delta_T, delta_C ]

	# Range of dS
	RdS = mxdS - mndS
	# Range coeff disp.
	RCD = mxCD - mnCD

	# Range of FE
	RFE = mxFE - mnFE

	print "F = ", L_FE_Dist[0:20]
	#print "FF = ", L_Dist[0:20]
	#print "RFE = ", RFE, RdS
	#cc = sys.stdin.read(1)

	# Now, stratify the distance in the simplex in nB bins
	InfoSimplex = {}

	# stratify the distance in FE
	InfoSimplexFE = {}

	# Stratify the simplex in nB bins, according to epsilon and lambda.
	InfoSimplexComb = {}

	if msamp == 'iid' or msamp == 'IID':
		nBsqrd = int (math.sqrt(nB) )
		# InfoSimplex[b] = [ dist, numgenes in bin, List of genes, average FE,  List of FE]
		numGinCentr = {}
		numGinCentrComb = {}

		# Number of genes in bin for FE
		numGinCentrFE = {}

		for g in G:
			ds = (GeneSimplex[g][0] - mndS) / RdS
			bb = int(nB * ds)
			if bb in numGinCentr:
				numGinCentr[bb] = numGinCentr[bb] + 1
			else:
				numGinCentr[bb] = 1
			if bb in InfoSimplex:
				InfoSimplex[bb][1] = InfoSimplex[bb][1] + 1
				InfoSimplex[bb][2].append(g)
				InfoSimplex[bb][3] = InfoSimplex[bb][3] + G[g][0]
				InfoSimplex[bb][4].append(G[g][0])
			else:
				#InfoSimplex[bb] = [ ds, 1, [g], G[g][0], [G[g][0]] ]
				dsx = GeneSimplex[g][0]
				InfoSimplex[bb] = [ dsx, 1, [g], G[g][0], [G[g][0]], cC ]
			GeneSimplex[g].append(bb)
			# The bin, obtained in terms of distance between g(T) and g(C)
			# and in terms of coefficient of dispersion in T
			dsE = ( GeneSimplex[g][6] - mnCD) / RCD
			# The bin for dispersion
			binDisp = int(dsE * nBsqrd)
			# The bin for distance
			binDist = int(ds * nBsqrd)

			# The pair of bins for the combined space
			GeneSimplex[g].append(binDisp)
			GeneSimplex[g].append(binDist)

			KK = numGinCentrComb.keys()
			v = is_in([binDisp,binDist], KK)
			if v == 1:
				numGinCentrComb[(binDisp,binDist)] = numGinCentrComb[(binDisp,binDist)] + 1
			else:
				numGinCentrComb[(binDisp,binDist)] = 1
			KK = InfoSimplexComb.keys()
			v = is_in([binDisp,binDist], KK)
			if v == 1:
				InfoSimplexComb[(binDisp, binDist)][1] = InfoSimplexComb[(binDisp, binDist)][1] + 1
				InfoSimplexComb[(binDisp, binDist)][2].append(g)
				InfoSimplexComb[(binDisp, binDist)][3] = InfoSimplexComb[(binDisp, binDist)][3] + G[g][0]
				InfoSimplexComb[(binDisp, binDist)][4].append(G[g][0])
			else:
				InfoSimplexComb[(binDisp, binDist)] = [dsx, 1, [g], G[g][0], [G[g][0]], cC ]

			# Here you are, AN
			# Clustering based on FE
			# GeneSimplex[g] = [dist in simplex, PosTrt, PosCtr, FE, totTreat, totCtr, delta_T, deltaa_C]
			# InfoSimplexFE[b] = [ dist, numgenes in bin, List of genes, average dist,  List of FE]
			dsFE = ( GeneSimplex[g][3] - mnFE) / RFE
			bb = int(nB * dsFE)
			if bb in numGinCentrFE:
				numGinCentrFE[bb] = numGinCentrFE[bb] + 1
			else:
				numGinCentrFE[bb] = 1
			if bb in InfoSimplexFE:
				InfoSimplexFE[bb][1] = InfoSimplexFE[bb][1] + 1
				InfoSimplexFE[bb][2].append(g)
				InfoSimplexFE[bb][3] = InfoSimplexFE[bb][3] + G[g][0]
				InfoSimplexFE[bb][4].append(G[g][0])
			else:
				#InfoSimplex[bb] = [ ds, 1, [g], G[g][0], [G[g][0]] ]
				dsx = GeneSimplex[g][0]
				InfoSimplexFE[bb] = [ dsFE, 1, [g], dsx, [dsx], cC ]
			GeneSimplex[g].append(bb)


		Cts = []
		CtsFE = []
		for b in range(nB+1):
			if b in InfoSimplex:
				Cts.append(InfoSimplex[b][0])
			else:
				Cts.append(0)
				numGinCentr[b] = 0
			if b in InfoSimplexFE:
				CtsFE.append(InfoSimplexFE[b][0])
			else:
				CtsFE.append(0)
				numGinCentrFE[b] = 0
		save_centroids(F_centroids, Cts, numGinCentr)
		save_centroids(F_centroidsFE, CtsFE, numGinCentrFE)
		# Combined structure
		Cts = []
		for b in range(nBsqrd+1):
			for c in range(nBsqrd+1):
				KK = InfoSimplexComb.keys()
				v = is_in([b, c], KK)
				if v == 1:
				#if [(b,c)] in InfoSimplexComb:
					Cts.append(InfoSimplexComb[(b,c)][0])
				else:
					Cts.append(0)
					numGinCentrComb[(b,c)] = 0
		print "Almost..."
		save_centroidsC(FC_centroids, nBsqrd, Cts, numGinCentrComb)
		print "saved"
	else:
		if msamp == 'kmeans' or msamp == 'km':
			nBFE = nB
			if autom == True:
			# Here you are, AN
			# Compute the error for kmeans from k = 1 to k = nB + 1
			# The error as a function of k should pinpoint the optimal number of bins:
			# Where the change in error stops decreasing
				# The error for kmeans for num bin = ii
				ErrnB = [0.0] * (nB + 1)
				nBnum = [None] * (nB + 1)

				ErrnBFE = [0.0] * (nB + 1)
				nBnumFE = [None] * (nB + 1)

				# The number of vectos in each bin, for each k
				for ii in range(nB+1):
					nBnum[ii] = [0] * (ii + 1)
					nBnumFE[ii] = [0] * (ii + 1)
				ln = len(L_Dist)
				for ii in range(2, nB + 1):
					k_means = cluster.KMeans(n_clusters = ii)
					k_means.fit(L_Dist)

					k_meansFE = cluster.KMeans(n_clusters = ii)
					k_meansFE.fit(L_FE_Dist)

					for jj in range(len(L_Dist)):
						lab = k_means.labels_[jj]
						dstE = abs(L_Dist[jj] - k_means.cluster_centers_[lab])[0]
						ErrnB[ii] = ErrnB[ii] + dstE
						nBnum[ii][lab] = nBnum[ii][lab] + 1

						labFE = k_meansFE.labels_[jj]
						#print "j = ", jj, labFE
						dstE_FE = abs(L_FE_Dist[jj] - k_meansFE.cluster_centers_[labFE])[0]
						#print "ds = ", dstE_FE, L_FE_Dist[jj], k_meansFE.cluster_centers_[labFE], lab
						ErrnBFE[ii] = ErrnBFE[ii] + dstE_FE
						nBnumFE[ii][labFE] = nBnumFE[ii][labFE] + 1

					ErrnB[ii] = ErrnB[ii] / float(ln)
					ErrnBFE[ii] = ErrnBFE[ii] / float(ln)
					print "ii = ", ii, ErrnB[ii], ErrnBFE[ii]
				nBopt = opt_numBin(ErrnB, nBnum)
				nBoptFE = opt_numBin(ErrnBFE, nBnumFE)
				nB = nBopt
				nBFE = nBoptFE
				print "opt nB = ", nB, nBopt, nBoptFE
				#cc = sys.stdin.read(1)

			nBsqrd = int (math.sqrt(nB) )
			numGinCentrComb = {}
		
			k_means = cluster.KMeans(n_clusters=nB+1)
			k_means.fit(L_Dist)
			Centers = []

			for i,c in enumerate(k_means.cluster_centers_):
				Centers.append([c, i])
			Cts_sort = sorted(Centers)

			#print "LFE = ", L_FE_Dist[0:50], L_FE_Dist[10000:10050]
			k_meansFE = cluster.KMeans(n_clusters=nBFE+1)
			k_meansFE.fit(L_FE_Dist)
			CentersFE = []

			for i,c in enumerate(k_meansFE.cluster_centers_):
				CentersFE.append([c, i])
			Cts_FE_sort = sorted(CentersFE)

			#print "Cts_FE_sort = ", Cts_FE_sort
			#cc = sys.stdin.read(1)

			k_meansDist = cluster.KMeans(n_clusters=nBsqrd+1)
			k_meansDist.fit(L_Dist)
			Centers_Dist = []
			for i,c in enumerate(k_meansDist.cluster_centers_):
				Centers_Dist.append([c, i])
			CtsDist_sort = sorted(Centers_Dist)

			k_meansDisp = cluster.KMeans(n_clusters=nBsqrd+1)
			k_meansDisp.fit(L_Disp)
			Centers_Disp = []
			for i,c in enumerate(k_meansDisp.cluster_centers_):
				Centers_Disp.append([c, i])
			CtsDisp_sort = sorted(Centers_Disp)

			numGinCentr = {}

			labDisp = {}
			labDist = {}

			# Number of genes in bin for FE
			numGinCentrFE = {}

			for ii,g in enumerate(G):
				lab = position(k_means.labels_[ii], Cts_sort)
				if lab in numGinCentr:
					numGinCentr[lab] = numGinCentr[lab] + 1
				else:
					numGinCentr[lab] = 1
				labFE = position(k_meansFE.labels_[ii], Cts_FE_sort)
				if labFE in numGinCentrFE:
					numGinCentrFE[labFE] = numGinCentrFE[labFE] + 1
				else:
					numGinCentrFE[labFE] = 1
				print "ii = ", ii
				print "a = ", g
				labDisp[g] = position(k_meansDisp.labels_[ii], CtsDisp_sort)
				labDist[g] = position(k_meansDist.labels_[ii], CtsDist_sort)

			for i,g in enumerate(G):
				ds = (GeneSimplex[g][0] - mndS) / RdS
				bb = position(k_means.labels_[i], Cts_sort)
				#print "bb = ", bb, k_means.labels_[i]
				#cc = sys.stdin.read(1)
				if bb in InfoSimplex:
					InfoSimplex[bb][1] = InfoSimplex[bb][1] + 1
					InfoSimplex[bb][2].append(g)
					InfoSimplex[bb][3] = InfoSimplex[bb][3] + G[g][0]
					InfoSimplex[bb][4].append(G[g][0])
				else:
					InfoSimplex[bb] = [ ds, 1, [g], G[g][0], [G[g][0]], cC ]
				GeneSimplex[g].append(bb)


				# The bin, obtained in terms of distance between g(T) and g(C)
				# and in terms of coefficient of dispersion in T
				"""
				dsE = ( GeneSimplex[g][6] - mnCD) / RCD
				# The bin for dispersion
				binDisp = int(dsE * nBsqrd)
				# The bin for distance
				binDist = int(ds * nBsqrd)
				"""
				binDisp = labDisp[g]
				binDist = labDist[g]
	
				# The pair of bins for the combined space
				GeneSimplex[g].append(binDisp)
				GeneSimplex[g].append(binDist)
	
				#print "bxx = ", binDisp, binDist, g, i
				KK = numGinCentrComb.keys()
				v = is_in([binDisp,binDist], KK)
				if v == 1:
				#if [binDisp,binDist] in numGinCentrComb:
					numGinCentrComb[(binDisp,binDist)] = numGinCentrComb[(binDisp,binDist)] + 1
				else:
					numGinCentrComb[(binDisp,binDist)] = 1
				KK = InfoSimplexComb.keys()
				v = is_in([binDisp,binDist], KK)
				if v == 1:
				#if [(binDisp,binDist)] in InfoSimplexComb:
					InfoSimplexComb[(binDisp, binDist)][1] = InfoSimplexComb[(binDisp, binDist)][1] + 1
					InfoSimplexComb[(binDisp, binDist)][2].append(g)
					InfoSimplexComb[(binDisp, binDist)][3] = InfoSimplexComb[(binDisp, binDist)][3] + G[g][0]
					InfoSimplexComb[(binDisp, binDist)][4].append(G[g][0])
				else:
					InfoSimplexComb[(binDisp, binDist)] = [ds, 1, [g], G[g][0], [G[g][0]], cC ]


				# Here you are, AN
				# Clustering based on FE
				# InfoSimplexFE[b] = [ dist, numgenes in bin, List of genes, average dist,  List of FE]
				dsFE = ( GeneSimplex[g][3] - mnFE) / RFE
				bb = position(k_meansFE.labels_[i], Cts_FE_sort)
				if bb in numGinCentrFE:
					numGinCentrFE[bb] = numGinCentrFE[bb] + 1
				else:
					numGinCentrFE[bb] = 1
				if bb in InfoSimplexFE:
					InfoSimplexFE[bb][1] = InfoSimplexFE[bb][1] + 1
					InfoSimplexFE[bb][2].append(g)
					InfoSimplexFE[bb][3] = InfoSimplexFE[bb][3] + GeneSimplex[g][0]
					InfoSimplexFE[bb][4].append(GeneSimplex[g][0])
				else:
					dsx = GeneSimplex[g][0]
					InfoSimplexFE[bb] = [ dsFE, 1, [g], dsx, [dsx], cC ]
				GeneSimplex[g].append(bb)


			Cts_f = []
			for i in Cts_sort:
				Cts_f.append(i[0][0])
			save_centroids(F_centroids, Cts_f, numGinCentr)

			CtsFE_f = []
			for i in Cts_FE_sort:
				print "i = ", i
				CtsFE_f.append(i[0][0])
			print "F = ", F_FE_centroids
			print "a = ", numGinCentrFE
			save_centroids(F_FE_centroids, CtsFE_f, numGinCentrFE)

			# Combined structure
			Cts_f = []
			for b in range(nBsqrd+1):
				for c in range(nBsqrd+1):
					KK = InfoSimplexComb.keys()
					v = is_in([b, c], KK)
					if v == 1:
					#if [(b,c)] in InfoSimplexComb:
						Cts_f.append(InfoSimplexComb[(b,c)][0])
					else:
						Cts_f.append(0)
						numGinCentrComb[(b,c)] = 0
			print "Almost..."
			save_centroidsC(FC_centroids, nBsqrd, Cts_f, numGinCentrComb)


		else:
			if msamp == 'random' or msamp == 'rand':
				# InfoSimplex[b] = [ dist, numgenes in bin, List of genes, average FE,  List of FE]
				numGinCentr = {}
				for aa in range(nB):
					numGinCentr[aa] = 0

				nBsqrd = int (math.sqrt(nB) )
				numGinCentrComb = {}

				for g in G:
					ds = (GeneSimplex[g][0] - mndS) / RdS
					bb = int(random.random()*nB)
					numGinCentr[bb] = numGinCentr[bb] + 1
					if bb in InfoSimplex:
						InfoSimplex[bb][1] = InfoSimplex[bb][1] + 1
						InfoSimplex[bb][2].append(g)
						InfoSimplex[bb][3] = InfoSimplex[bb][3] + G[g][0]
						InfoSimplex[bb][4].append(G[g][0])
					else:
						#InfoSimplex[bb] = [ ds, 1, [g], G[g][0], [G[g][0]] ]
						dsx = GeneSimplex[g][0]
						InfoSimplex[bb] = [ dsx, 1, [g], G[g][0], [G[g][0]], cC ]
					GeneSimplex[g].append(bb)


					# The bin, obtained in terms of distance between g(T) and g(C)
					# and in terms of coefficient of dispersion in T
					"""
					dsE = ( GeneSimplex[g][6] - mnCD) / RCD
					# The bin for dispersion
					binDisp = int(dsE * nBsqrd)
					# The bin for distance
					binDist = int(ds * nBsqrd)
					"""
					binDisp = int(random.random()*nBsqrd)
					binDist = int(random.random()*nBsqrd)
		
					# The pair of bins for the combined space
					GeneSimplex[g].append(binDisp)
					GeneSimplex[g].append(binDist)
		
					print "bxx = ", binDisp, binDist, g
					KK = numGinCentrComb.keys()
					v = is_in([binDisp,binDist], KK)
					if v == 1:
					#if [binDisp,binDist] in numGinCentrComb:
						numGinCentrComb[(binDisp,binDist)] = numGinCentrComb[(binDisp,binDist)] + 1
					else:
						numGinCentrComb[(binDisp,binDist)] = 1
					KK = InfoSimplexComb.keys()
					v = is_in([binDisp,binDist], KK)
					if v == 1:
					#if [(binDisp,binDist)] in InfoSimplexComb:
						InfoSimplexComb[(binDisp, binDist)][1] = InfoSimplexComb[(binDisp, binDist)][1] + 1
						InfoSimplexComb[(binDisp, binDist)][2].append(g)
						InfoSimplexComb[(binDisp, binDist)][3] = InfoSimplexComb[(binDisp, binDist)][3] + G[g][0]
						InfoSimplexComb[(binDisp, binDist)][4].append(G[g][0])
					else:
						InfoSimplexComb[(binDisp, binDist)] = [ds, 1, [g], G[g][0], [G[g][0]], cC ]


				Cts = []
				for b in range(nB+1):
					if b in InfoSimplex:
						Cts.append(InfoSimplex[b][0])
					else:
						Cts.append(0)
						numGinCentr[b] = 0
				save_centroids(F_centroids, Cts, numGinCentr)

				# Combined structure
				Cts = []
				for b in range(nBsqrd+1):
					for c in range(nBsqrd+1):
						KK = InfoSimplexComb.keys()
						v = is_in([b, c], KK)
						if v == 1:
						#if [(b,c)] in InfoSimplexComb:
							Cts.append(InfoSimplexComb[(b,c)][0])
						else:
							Cts.append(0)
							numGinCentrComb[(b,c)] = 0
				print "Almost..."
				save_centroidsC(FC_centroids, nBsqrd, Cts, numGinCentrComb)


	# Obtain the average FE
	for b in range(nB+1):
		#print "b = ", b
		if b in InfoSimplex:
			InfoSimplex[b][3] = InfoSimplex[b][3]/InfoSimplex[b][1]
		else:
			InfoSimplex[b] = [0.0, 0, [], 0.0, []]

	for b in range(nBFE+1):
		#print "b = ", b
		if b in InfoSimplexFE:
			InfoSimplexFE[b][3] = InfoSimplexFE[b][3]/InfoSimplexFE[b][1]
		else:
			InfoSimplexFE[b] = [0.0, 0, [], 0.0, []]

	return [GeneSimplex, InfoSimplex, InfoSimplexComb, InfoSimplexFE]


def mad_based_outlier(points, thresh):
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return [modified_z_score > thresh, median, modified_z_score]

def	save_genes(FF, L, GeneB, Zs, An, GS, G, nn):
	#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]
	#GeneB[g] = [b, fe, avgF, numG]
	#G[g] = [fe, Trt, Ctr, totTrt, totCtr]
	#Zs[g] is the mod Z score
	#print "GB = ", GeneB
	#print "L = ", L
	f = open(FF, "w")
	f.write("#gene\tbin\tdistS\t")
	for t in range(nn):
		f.write("T_" + str(t) + "\t")
	for c in range(nn):
		f.write("C_" + str(c) + "\t")
	f.write("FE\tavgFE\tnumG\ttotTrt\ttotCtr\tScore\tAnom\n")
	for g in L:
		#print "g = ", g
		#print "GB = ", GeneB[g]
		#print "Z = ", Zs[g]
		f.write(g + "\t" + str(GeneB[g][0]) + "\t" + str(GS[g][0]) + "\t")
		if len(GeneB[g]) == 5:
			f.write(str(GeneB[g][4]) + "\t")
		for t in G[g][1]:
			f.write( str(t) + "\t")
		for c in G[g][2]:
			f.write( str(c) + "\t")
		f.write(str(GeneB[g][1]) + "\t" + str(GeneB[g][2]) + "\t" + str(GeneB[g][3]) + "\t" + str(GS[g][4]) + "\t" + str(GS[g][5]) + "\t" + str(Zs[g])  + "\t" + str(An[g]) + "\n")
	f.close()

def	save_genesC(FF, L, GeneBComb, Zs, An, GS, G, nn):
	#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]
	#GeneB[g] = [b, fe, avgF, numG]
	#G[g] = [fe, Trt, Ctr, totTrt, totCtr]
	#Zs[g] is the mod Z score
	#print "GB = ", GeneB
	#print "L = ", L
	f = open(FF, "w")
	f.write("#gene\tbDisp\tbDist\tdistS\t")
	for t in range(nn):
		f.write("T_" + str(t) + "\t")
	for c in range(nn):
		f.write("C_" + str(c) + "\t")
	f.write("FE\tavgFE\tnumG\ttotTrt\ttotCtr\tScore\tAnom\n")
	for g in L:
		#print "g = ", g
		#print "GB = ", GeneB[g]
		#print "Z = ", Zs[g]
		f.write(g + "\t" + str(GeneBComb[g][0]) + "\t" + str(GeneBComb[g][1]) +  "\t" + str(GS[g][0]) + "\t")
		for t in G[g][1]:
			f.write( str(t) + "\t")
		for c in G[g][2]:
			f.write( str(c) + "\t")
		f.write(str(GeneBComb[g][2]) + "\t" + str(GeneBComb[g][3]) + "\t" + str(GeneBComb[g][4]) + "\t" + str(GS[g][4]) + "\t" + str(GS[g][5]) + "\t" + str(Zs[g])  + "\t" + str(An[g]) + "\n")
	f.close()


def	save_genes_total(FF, GeneL, Zs_all, GeneSimplex, G, nn):
#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]
	f = open(FF, "w")
	for g in GeneL:
		f.write(g + "\t" + str(GeneSimplex[g][0]) + "\t" + str(GeneSimplex[g][3]) + "\t" + str(Zs_all[g]) + "\n")
	f.close()

"""
the PDF file is obtained from:

cat Emmi_THP1_DE_CDA_2.csv | awk 'BEGIN{ N = 100; } {X = $4; v = int(N*(X/1.41421)); PDF[v] = PDF[v] + 1; G[v] = G[v]""$1"\t"; if($9 != 0){fe = log($9)/log(10); avgR[v] = avgR[v] + fe; PDFno0[v] = PDFno0[v] + 1; Gno0[v] = Gno0[v]""$1" "fe"\t"; }; }    END{ T = 0; cumPDFno0 = 0; for(i = 0; i <= N; i++){ if(PDFno0[i] > 0){R = avgR[i]/PDFno0[i];  cumPDFno0 = cumPDFno0 + PDFno0[i]; } else{R = 0}; d = 2*i/(1.41421*N); print i"\t"d"\t"PDF[i]"\t"PDFno0[i]"\t"avgR[i]"\t"cumPDFno0"\t"Gno0[i]"\t---\t"G[i]; T = T + PDF[i]}; print "#Tot:\t"T }' > PDF_Emmi_THP1_DE_CDA_2_FE.csv


"""
parser = argparse.ArgumentParser()


parser.add_argument('-i', action = "store", dest = "i", help = "The file containing the abundance of relevant genes for the two cases under comparison")
parser.add_argument('-n', action = "store", dest = "n", help = "The number of repeats (replicates) for each case (So far the same number for both conditions is required)")
#parser.add_argument('-i', action = "store", dest = "i", help = "The file containing the PDF info")
parser.add_argument('-th', action = "store", dest = "th", help = "The threshold for MAD")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file containing the bin info etc")
parser.add_argument('-oC', action = "store", dest = "oC", help = "The output file for the combined case, containing the bin info etc")
parser.add_argument('-r', action = "store", dest = "r", help = "The output file containing the list of relevant genes")
parser.add_argument('-rC', action = "store", dest = "rC", help = "The output file for the combined case, containing the list of relevant genes")
parser.add_argument('-k', action = "store", dest = "k", help = "The proportion of vectors to compare to")
parser.add_argument('-kk', action = "store", dest = "kk", help = "The proportion of vectors to compare to in LOF (simplex space)")
parser.add_argument('-th1', action = "store", dest = "th1", help = "The lower limit for lof to be considred an anomaly")
parser.add_argument('-th2', action = "store", dest = "th2", help = "The upper limit for lof to be considred an anomaly")
parser.add_argument('-a', action = "store", dest = "a", help = "The anomaly detection algorithm")
parser.add_argument('-s', action = "store", dest = "s", help = "The output file containing the sorted list of genes")
parser.add_argument('-sC', action = "store", dest = "sC", help = "The output file for the combined case, containing the sorted list of genes")
parser.add_argument('-ne', action = "store", dest = "ne", help = "The number of estimators for isolation forests")
parser.add_argument('-c', action = "store", dest = "c", help = "The output file containing the anomaly metric when genes are considered in the space {dist(g(t), g(c)), FE(g)}")
parser.add_argument('-cc', action = "store", dest = "cc", help = "The output file containing the anomaly metric when genes are considered in the space {dist(g(t), g(c)), FE(g)}")
parser.add_argument('-p', action = "store", dest = "p", help = "The image file containing the scatter plot of all considered genes in the space {(g(t), g(c)), FE(g)}")
parser.add_argument('-m', action = "store", dest = "m", help = "The method for definying the  distribution of bins")
parser.add_argument('-autom', action = "store_true", dest = "autom", help = "Detect the optimal number of bins")
parser.add_argument('-nB', action = "store", dest = "nB", help = "The number of bins in which the distance in the simplex is to be divided into")
parser.add_argument('-kc', action = "store", dest = "kc", help = "The centroids if -m is kmeans")
parser.add_argument('-kcC', action = "store", dest = "kcC", help = "The combined centroids if -m is kmeans")

args = parser.parse_args()

nn = int(args.n)
G = read_data(args.i, nn)
# G[g] = [fe, Trt, Ctr, totTrt, totCtr, dist(T,C)]

nB = int(args.nB)
#nB = 100

GeneSimplex, InfoSimplex, InfoSimplexComb, InfoSimplexFE = create_simplex(nn, G, nB, args.m, args.kc, args.kcC, args.autom)
#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, delta_T, delta_C, bin, binDisp, binDist, binFE]
# InfoSimplex[b] = [ dist, numgenes in bin, List of genes, average FE,  List of FE, coordinates centre]
# InfoSimplexComb[(b,c)] = [ dist, numgenes in bin, List of genes, average FE,  List of FE, coordinates centre]

# BinGen[b] is the list of genes in bin b 
# BinFE[b] is the list of FE for genes in bin b 
# BinL is the list of bins
# DistSimplexBin[b] is the distance (between T and C cases) associated to bin b
# BinNumG[b] is the number of genes in bin b
# avgBinFE[b] is the average FE for bin b
# GeneB[g] contains [b, fe, avgF, numG]
# where b is the bin, fe its fe, avgF the average FE for genes in bin b, and
# numG is the number of genes in bin b

BinGen = {}
BinFE = {}
DistSimplexBin = {}
BinNumG = {}
avgBinFE = {}

# The anomaly is computed over the dispersion (distance between g(T) and g(C)
# BinGenCDA[b] is the list of genes in bin b 
# BinCDA[b] is the list of d(T,C) for genes in bin b 
# BinLFE is the list of bins
# DistSimplexBinFE[b] is the distance (between T and C cases) associated to bin b
# BinNumGFE[b] is the number of genes in bin b
# avgBinFE[b] is the average FE for bin b
# GeneB[g] contains [b, fe, avgF, numG]
# where b is the bin, fe its fe, avgF the average FE for genes in bin b, and
# numGFE is the number of genes in bin b
BinCDA = {}
BinGenFE = {}
DistSimplexBinFE = {}
BinNumGFE = {}
avgBinCDA = {}

for b in InfoSimplex:
	BinGen[b] = InfoSimplex[b][2]
	BinFE[b] = InfoSimplex[b][4]
	DistSimplexBin[b] = InfoSimplex[b][0]
	BinNumG[b] = InfoSimplex[b][1]
	avgBinFE[b] = InfoSimplex[b][3]
BinL = InfoSimplex.keys()

# InfoSimplexFE[b] = [ dist, numgenes in bin, List of genes, average dist,  List of FE]
for b in InfoSimplexFE:
	BinGenFE[b] = InfoSimplexFE[b][2]
	BinCDA[b] = InfoSimplexFE[b][4]
	DistSimplexBinFE[b] = InfoSimplexFE[b][0]
	BinNumGFE[b] = InfoSimplexFE[b][1]
	avgBinCDA[b] = InfoSimplexFE[b][3]

BinLFE = InfoSimplexFE.keys()
GeneBFE = {}

GeneB = {}
GeneBComb = {}

for g in GeneSimplex:
	#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, delta_T, delta_C, bin, binDisp, binDist, binFE]
	#b: the bin in which gene g i located in the simplex
	b = GeneSimplex[g][8]
	GeneB[g] = [ b, GeneSimplex[g][3], InfoSimplex[b][3], InfoSimplex[b][1] ]
	# GeneB[g] contains [b, fe, avgF, numG]

	# InfoSimplexFE[b] = [ dist, numgenes in bin, List of genes, average dist,  List of FE]
	bFE = GeneSimplex[g][11]
	GeneBFE[g] = [ bFE, GeneSimplex[g][3], InfoSimplexFE[bFE][3], InfoSimplexFE[bFE][1], G[g][5] ] 
	#GeneBFE[g] = [ bFE, GeneSimplex[g][0], InfoSimplexFE[bFE][3], InfoSimplexFE[bFE][1], G[g][5] ] 
	# GeneBFE[g] contains [bFE, d(T,C), avgDst, numG]

	#print "h = ", GeneSimplex[g]

	#print "G = ", GeneSimplex[g][3]
	bDisp = GeneSimplex[g][9]
	bDist = GeneSimplex[g][10]
	#print "k = ", InfoSimplexComb[(bDisp,bDist)]
	GeneBComb[g] = [ bDisp, bDist, GeneSimplex[g][3], InfoSimplexComb[(bDisp,bDist)][3], InfoSimplexComb[(bDisp,bDist)][1] ]


T = [17]

CDA_G = []
CDA_G_all = []
MAD_Anom = {}

CDA_G_FE = []
CDA_G_FE_all = []
MADFE_Anom = {}

Zs = {}
Zs_all = {}

ZsFE = {}
ZsFE_all = {}

An = {}
AnFE = {}

print "moving further"

#for b in T:
#for b in BinL:
for b in range(nB):
	#print "b = ", b
	#print "FE = ", len(BinFE[b])
	if args.a == '0' or args.a == 'MAD' or args.a == 'mad':
		Th = float(args.th)
		[Anom, Med, modZscore] = mad_based_outlier(np.array(BinFE[b]), Th)
		AnomMetric = list(modZscore)
		[AnomFE, MedFE, modZscoreFE] = mad_based_outlier(np.array(BinCDA), Th)
		AnomMetricFE = list(modZscoreFE)
	else:
		if args.a == '1' or args.a == 'lof' or args.a == 'LOF':
			Anom = []
			if b in BinL:
				ln = len(BinFE[b])
				if ln > 1:
					AnomMetric = []
					Th1 = float(args.th1)
					Th2 = float(args.th2)
					k = float(args.k)
					numK = int(ln * k) + 1
					if numK >= int(ln/5):
						numK = int(ln/5) + 1
					#print "nK = ", numK, ln
					D = []
					for ff in BinFE[b]:
						D.append(tuple([ff]))
					#print "DL = ", D
					#print "b = ", b
					clf = LOF(n_neighbors=numK)
					clf.fit(D)
					y_train_pred = clf.labels_
					y_train_scores = clf.decision_scores_
					for idx, s in enumerate(D):
						if y_train_scores[idx] <= Th1 or y_train_scores[idx] >= Th2:
							Anom.append(True)
						else:
							Anom.append(False)
						AnomMetric.append(y_train_scores[idx])
			# Transpose (clustered by FE)
			AnomFE = []
			if b in BinLFE:
				ln = len(BinCDA[b])
				if ln > 1:
					AnomMetricFE = []
					Th1 = float(args.th1)
					Th2 = float(args.th2)
					k = float(args.k)
					numK = int(ln * k) + 1
					if numK >= int(ln/5):
						numK = int(ln/5) + 1
					D = []
					for ff in BinCDA[b]:
						D.append(tuple([ff]))
					clf = LOF(n_neighbors=numK)
					clf.fit(D)
					y_train_pred = clf.labels_
					y_train_scores = clf.decision_scores_
					for idx, s in enumerate(D):
						if y_train_scores[idx] <= Th1 or y_train_scores[idx] >= Th2:
							AnomFE.append(True)
						else:
							AnomFE.append(False)
						AnomMetricFE.append(y_train_scores[idx])
		else:
			if args.a == '2' or args.a == 'if' or args.a == 'IF':
				ln = len(BinFE[b])
				if ln > 1:
					#nK[b] = -1
					numE = int(args.ne)
					Th = float(args.th)
					D = []
					Anom = []
					AnomMetric = []
					for ff in BinFE[b]:
						D.append(tuple([ff]))
					clf = IForest(n_estimators = numE)
					#clf = IForest()
					clf.fit(D)
					y_train_pred = clf.labels_
					y_train_scores = clf.decision_scores_
					for idx, s in enumerate(D):
						"""
						if y_train_pred[idx] == 0:
							Anom.append(False)
						else:
							Anom.append(True)
						"""
						if y_train_scores[idx] >= Th:
							Anom.append(True)
						else:
							Anom.append(False)
						AnomMetric.append(y_train_scores[idx])
				else:
					Anom = []
					algo = True
	X_Anom = []
	Y_Anom = []
	G_Anom = []
	#print "Anom = ", Anom
	for i,l in enumerate(Anom):
		#print "i = ", i, l, len(Anom), b, BinGen[b]
		g = BinGen[b][i]
		if l == True:
			X_Anom.append(i)
			#g = BinGen[b][i]
			CDA_G.append(g)
			#print "g = ", g
			Y_Anom.append(GeneB[g][1])
			G_Anom.append(g)
			Zs[g] = AnomMetric[i]
			Zs_all[g] = AnomMetric[i]
			An[g] = "yes"
		else:
			X_Anom.append(i)
			Y_Anom.append(0)
			G_Anom.append('')
			Zs_all[g] = AnomMetric[i]
			An[g] = "no"
		CDA_G_all.append(g)
	if b in BinL:
		X = np.linspace(0,BinNumG[b]-1, BinNumG[b], dtype = int)
		plt.scatter(X, BinFE[b], alpha=0.5)

		i = 0
		for x,y in zip(X_Anom,Y_Anom):
			label = G_Anom[i]
			plt.annotate(label, (x,y))# this is the text
			i = i + 1
		plt.savefig(args.o + "_RV_" + str(b) + ".png")

		plt.clf()

		sns.distplot(BinFE[b])
		#sns.distplot(BinFE[b], bins=int(len(BinFE[b])/20))
		plt.savefig(args.o + "_PDF_" + str(b) + ".png")
		plt.clf()

	# Transpose (clusterd by FE)
	XFE_Anom = []
	YFE_Anom = []
	GFE_Anom = []
	#print "Anom = ", Anom
	for i,l in enumerate(AnomFE):
		#print "i = ", i, l, len(Anom), b, BinGen[b]
		g = BinGenFE[b][i]
		if l == True:
			XFE_Anom.append(i)
			#g = BinGen[b][i]
			CDA_G_FE.append(g)
			#print "g = ", g
			YFE_Anom.append(GeneBFE[g][1])
			GFE_Anom.append(g)
			ZsFE[g] = AnomMetricFE[i]
			ZsFE_all[g] = AnomMetricFE[i]
			AnFE[g] = "yes"
		else:
			XFE_Anom.append(i)
			YFE_Anom.append(0)
			GFE_Anom.append('')
			ZsFE_all[g] = AnomMetricFE[i]
			AnFE[g] = "no"
		CDA_G_FE_all.append(g)
	if b in BinLFE:
		XFE = np.linspace(0,BinNumGFE[b]-1, BinNumGFE[b], dtype = int)
		plt.scatter(XFE, BinCDA[b], alpha=0.5)

		i = 0
		for x,y in zip(XFE_Anom,YFE_Anom):
			label = GFE_Anom[i]
			plt.annotate(label, (x,y))# this is the text
			i = i + 1
		plt.savefig(args.o + "_RV_" + str(b) + "_Transpose.png")
		#plt.show()

		plt.clf()

		sns.distplot(BinCDA[b])
		plt.savefig(args.o + "_PDF_" + str(b) + "_Transpose.png")
		plt.clf()


#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]
save_genes(args.r, CDA_G, GeneB, Zs, An, GeneSimplex, G, nn)
save_genes(args.s, CDA_G_all, GeneB, Zs_all, An, GeneSimplex, G, nn)

save_genes(args.r + "_Transpose", CDA_G_FE, GeneBFE, ZsFE, AnFE, GeneSimplex, G, nn)
save_genes(args.s + "_Transpose", CDA_G_FE_all, GeneBFE, ZsFE_all, AnFE, GeneSimplex, G, nn)



CDA_G_all = []

print "Second stage"
# Now, detect genes with different expression as anomalies in the space
# { dist(g(t), g(c)), FE(g)}

#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]
# InfoSimplex[b] = [ dist, numgenes in bin, List of genes, average FE,  List of FE]
spaceSimplex = []
GeneL = []
Zs = {}
Zs_all = {}
for g in GeneSimplex:
	spaceSimplex.append( [GeneSimplex[g][0], GeneSimplex[g][3]] )
	GeneL.append(g)

print "l = ", len(GeneL)

if args.a == '0' or args.a == 'MAD' or args.a == 'mad':
	Th = float(args.th)
	[Anom, Med, modZscore] = mad_based_outlier(np.array(spaceSimplex), Th)
	AnomMetric = list(modZscore)
else:
	if args.a == '1' or args.a == 'lof' or args.a == 'LOF':
		Anom = []
		ln = len(spaceSimplex)
		if ln > 1:
			#Anom = []
			AnomMetric = []
			Th1 = float(args.th1)
			Th2 = float(args.th2)
			k = float(args.kk)
			numK = int(ln * k) + 1
			if numK >= int(ln/5):
				numK = int(ln/5)
			#print "nK = ", numK, ln
			D = []
			for ff in spaceSimplex:
				D.append(tuple([ff]))
			clf = LOF(n_neighbors=numK)
			clf.fit(spaceSimplex)
			#clf.fit(D)
			y_train_pred = clf.labels_
			y_train_scores = clf.decision_scores_
			for idx, s in enumerate(D):
				"""
				if y_train_pred[idx] == 0:
					Anom.append(False)
				else:
					Anom.append(True)
				"""
				if y_train_scores[idx] <= Th1 or y_train_scores[idx] >= Th2:
					Anom.append(True)
				else:
					Anom.append(False)
				AnomMetric.append(y_train_scores[idx])
	else:
		if args.a == '2' or args.a == 'if' or args.a == 'IF':
			#nK[b] = -1
			numE = int(args.ne)
			Th = float(args.th)
			D = []
			Anom = []
			AnomMetric = []
			for ff in spaceSimplex:
				D.append(tuple([ff]))
			clf = IForest(n_estimators = numE)
			#clf = IForest()
			clf.fit(spaceSimplex)
			#clf.fit(D)
			y_train_pred = clf.labels_
			y_train_scores = clf.decision_scores_
			for idx, s in enumerate(D):
				"""
				if y_train_pred[idx] == 0:
					Anom.append(False)
				else:
					Anom.append(True)
				"""
				if y_train_scores[idx] >= Th:
					Anom.append(True)
				else:
					Anom.append(False)
				AnomMetric.append(y_train_scores[idx])

print "AD done"


X_Anom = []
Y_Anom = []
G_Anom = []
print "Anom = ", len(Anom)
for i,l in enumerate(Anom):
	#print "i = ", i, l, len(Anom)
	g = GeneL[i]
	if l == True:
		X_Anom.append(i)
		CDA_G.append(g)
		Y_Anom.append(GeneSimplex[g][0])
		G_Anom.append(g)
		Zs[g] = AnomMetric[i]
		Zs_all[g] = AnomMetric[i]
		An[g] = "yes"
	else:
		X_Anom.append(i)
		Y_Anom.append(0)
		G_Anom.append('')
		Zs_all[g] = AnomMetric[i]
		An[g] = "no"
	CDA_G_all.append(g)
print "advancing"
spA = np.array(spaceSimplex)
plt.scatter( spA[:,0], spA[:,1], alpha=0.5)
i = 0
for x,y in zip(X_Anom,Y_Anom):
	label = G_Anom[i]
	#print "l = ", label
	plt.annotate(label, (x,y))# this is the text
	i = i + 1
print "Plotting"
#plt.scatter(X, spaceSimplex, s=area, c=colors, alpha=0.5)
plt.savefig(args.p)
#plt.show()

plt.clf()

save_genes_total(args.c, GeneL, Zs_all, GeneSimplex, G, nn)
save_genes(args.cc, CDA_G_all, GeneB, Zs_all, An, GeneSimplex, G, nn)
#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]


print "Third stage"
"""
Characterize FE(g) = E(g,T)/E(g,C) for similar genes.
The third stage is considering the space {delta(g), epsilon_T(g,T), epsilon_C(g)} for such
characterization.

delta(g) is the distance between g(T) and g(C)
g(T) is the position of gene g in the composition T
g(C) is the position of gene g in the composition C

epsilon_T(g) is the distance between g(T) and c
epsilon_C(g) is the distance between g(C) and c
c is the center of a n-sided composition, with coordinates {1/n,1/n,...,1/n}
"""

BinFE = {}
BinGen = {}
DistSimplexBin = {}
BinNumG = {}
avgBinFE = {}

for b in InfoSimplexComb:
	BinGen[b] = InfoSimplexComb[b][2]
	BinFE[b] = InfoSimplexComb[b][4]
	DistSimplexBin[b] = InfoSimplexComb[b][0]
	BinNumG[b] = InfoSimplexComb[b][1]
	avgBinFE[b] = InfoSimplexComb[b][3]

BinL = InfoSimplexComb.keys()

CDA_G = []
CDA_G_all = []
MAD_Anom = {}

Zs = {}
Zs_all = {}

An = {}

#for b in T:
for b in BinL:
	print "b = ", b
	print "FE = ", len(BinFE[b])
	if args.a == '0' or args.a == 'MAD' or args.a == 'mad':
		Th = float(args.th)
		[Anom, Med, modZscore] = mad_based_outlier(np.array(BinFE[b]), Th)
		AnomMetric = list(modZscore)
	else:
		if args.a == '1' or args.a == 'lof' or args.a == 'LOF':
			Anom = []
			ln = len(BinFE[b])
			if ln > 1:
			#if ln > 1:
				#Anom = []
				AnomMetric = []
				Th1 = float(args.th1)
				Th2 = float(args.th2)
				k = float(args.k)
				numK = int(ln * k) + 1
				if numK >= int(ln/5):
					numK = int(ln/5) + 1
				#print "nK = ", numK, ln
				D = []
				for ff in BinFE[b]:
					D.append(tuple([ff]))
				#print "DL = ", D
				#print "b = ", b
				clf = LOF(n_neighbors=numK)
				clf.fit(D)
				y_train_pred = clf.labels_
				y_train_scores = clf.decision_scores_
				for idx, s in enumerate(D):
					"""
					if y_train_pred[idx] == 0:
						Anom.append(False)
					else:
						Anom.append(True)
					"""
					if y_train_scores[idx] <= Th1 or y_train_scores[idx] >= Th2:
						Anom.append(True)
					else:
						Anom.append(False)
					AnomMetric.append(y_train_scores[idx])
		else:
			if args.a == '2' or args.a == 'if' or args.a == 'IF':
				ln = len(BinFE[b])
				if ln > 1:
					#nK[b] = -1
					numE = int(args.ne)
					Th = float(args.th)
					D = []
					Anom = []
					AnomMetric = []
					for ff in BinFE[b]:
						D.append(tuple([ff]))
					clf = IForest(n_estimators = numE)
					#clf = IForest()
					clf.fit(D)
					y_train_pred = clf.labels_
					y_train_scores = clf.decision_scores_
					for idx, s in enumerate(D):
						"""
						if y_train_pred[idx] == 0:
							Anom.append(False)
						else:
							Anom.append(True)
						"""
						if y_train_scores[idx] >= Th:
							Anom.append(True)
						else:
							Anom.append(False)
						AnomMetric.append(y_train_scores[idx])
				else:
					Anom = []
					algo = True
	X_Anom = []
	Y_Anom = []
	G_Anom = []
	#print "Anom = ", Anom
	for i,l in enumerate(Anom):
		#print "i = ", i, l, len(Anom), b, BinGen[b]
		g = BinGen[b][i]
		if l == True:
			X_Anom.append(i)
			#g = BinGen[b][i]
			CDA_G.append(g)
			#print "g = ", g
			Y_Anom.append(GeneBComb[g][2])
			G_Anom.append(g)
			Zs[g] = AnomMetric[i]
			Zs_all[g] = AnomMetric[i]
			An[g] = "yes"
		else:
			X_Anom.append(i)
			Y_Anom.append(0)
			G_Anom.append('')
			Zs_all[g] = AnomMetric[i]
			An[g] = "no"
		CDA_G_all.append(g)
	X = np.linspace(0,BinNumG[b]-1, BinNumG[b], dtype = int)
	plt.scatter(X, BinFE[b], alpha=0.5)
	#print "X = ", X_Anom, len(X_Anom)
	#print "Y = ", Y_Anom, len(Y_Anom)
	#print "G = ", G_Anom, len(G_Anom)
	i = 0
	for x,y in zip(X_Anom,Y_Anom):
		label = G_Anom[i]
		#print "l = ", label
		plt.annotate(label, (x,y))# this is the text
		i = i + 1
#                 (x,y), # this is the point to label
#                 textcoords="offset points", # how to position the text
#                 xytext=(0,10), # distance from text to points (x,y)
#                 ha='center') # horizontal alignment can be left, right or center
#	plt.text([3, 4, 5, 9], [1.0, 2.5, 3.5, 2.1], ['a', 'b', 'c', 'd'])
	#plt.text(X_Anom, Y_Anom, '%s' % G_Anom)
	#plt.scatter(X_Anom, Y_Anom, alpha=0.5, c = 'red')

	#plt.scatter(X, BinFE[b], s=area, c=colors, alpha=0.5)
	plt.savefig(args.oC + "_RV_" + str(b) + ".png")
	#plt.show()

	plt.clf()

	sns.distplot(BinFE[b])
	#sns.distplot(BinFE[b], bins=int(len(BinFE[b])/20))
	plt.savefig(args.oC + "_PDF_" + str(b) + ".png")
	plt.clf()

#GeneSimplex[g] = [dS, PosTrt, PosCtr, fe, totTreat, totCtr, bin]
save_genesC(args.rC, CDA_G, GeneBComb, Zs, An, GeneSimplex, G, nn)
save_genesC(args.sC, CDA_G_all, GeneBComb, Zs_all, An, GeneSimplex, G, nn)


