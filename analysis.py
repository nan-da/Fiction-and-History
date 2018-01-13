from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, dendrogram
from pandas import DataFrame, Series
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
import numpy as np
import random, pickle
from sklearn.preprocessing import scale
import paulutility

###################
# PARAMETER SETUP #
###################

# This section of code changes the various parameters on which the PCA or HCA runs

# Name for the resultant pdf
savefigurename = "results.pdf"
screeplotname = "scree_"+savefigurename

#pcs to plot. note, this is pc number - 1 (as the pcs are zero indexed)
pcs = [0, 1]

# even number of texts?
textbalance = False

# List of items to which to limit the analysis
# 0 for genre, 1 for author
item_type = 0
items_to_include = ["正史","小说", "演义", "野史"]

# limit to specific eras
eras = None #["明", "清"]

# vocab (None will include everything. Otherwise, use a list of terms)
voc = None
# Set to "yes" to add labels to the PCA
wanttoannottate = "no"
annottate_font_size = 8

# Run PCA or HCA
analysisoptions = ["pca", "hca"]
analysischoise = analysisoptions[0]

# Run on full length texts, split into chapter/juan, or split texts in to
# sections of equal length
full_split = ["full", "split", "num"]
fschoise = full_split[0]

# If splitting in to equal lengths, how long should the section be?
if fschoise == "num":
	divnum = 10000

# How should the ngram counts be normalized?
# TFIDF, term frequency, raw counts, and a chi-based measure (Ted Underwood's)
countnormtype = ["tfidf", "tf", "raw", "chi"]
cnchoice = "tf" #default "tf"

# How many features should be analyzed?
features = 1000

# For HCA, what distance/similarity measure should be used?
distance_measures = ["euc", "cos"]
distchoice = distance_measures[0]

# Set the alpha (transparency) for the resultant graph
alfa = 1

# Specify an ngram range in the form of a tuple. First number is smallest ngram
# second is largest
ngramrange = (1,1)

# Name of directories to scan for files
# If you are using this with dataverse, you will need to
# put the files to analyze in a folder of your choosing in the same
# directory as this python file. This folder will also
# need to contain a "metadata.txt" file. The metadata file
# needs to be a .tsv with the filename, genre, author, title, era columns
directories = ["corpus"]



# Size of resultant PDF
figsize = (10,10)

# Colors and labels for HCA/PCA
# 1 is title, 2 is genre, 3 is era, 4 is author, 5 is dir, 6 is secname
label_name = 2
label_color = 2

# Size of the dots in the PCA
dot_size=4

# Set mean to zero?
scalemeanzero = False

# Plot Loadings?
plotloadings = False


##################
# PICKLE THINGS? #
##################

# Save a pickle of the PCA loadings?
pickle_loadings = True

''' Unimplemented features
#Pickle or Analyze?
topickle = False

if topickle == True:
	skipanalysis = True:
elif topickle == False:
	skipanalysis = False:

skipimport = False




# First number 1 used or 0 unused, second min features, third max featueres
featurerange = (0, 0, 400)

# Set Vocabulary 1 used 0 unused, list of vocab
setvocab = (0, [])

'''

################
# MAIN PROGRAM #
################
if __name__ == "__main__":

	# Get yeshi titles
	y1 = open("ymetadata.txt","r").read()
	yeshititles = []

	for line in y1.split("\n"):
		cells = line.split("\t")
		yeshititles.append(cells[2])




	print("Acquiring Data")
	# Send the directories and genres to include to the appropriate function
	# from paulutility.
	if fschoise == "full":
		infolist = paulutility.fulltextcontent(directories, items_to_include, item_type, e=eras)
	elif fschoise == "split":
		infolist = paulutility.splittextcontent(directories, items_to_include, item_type,e=eras)
	elif fschoise == "num":
		infolist = paulutility.fullsplitbynum(directories, divnum, items_to_include, item_type, e=eras)
	else:
		print("Not a valid choice")
		# Kill the program
		exit()

	for i,title in enumerate(infolist[1]):
		if title in yeshititles:
			infolist[2][i] = "野史"


	'''
		if "外史" in title or "逸史" in title or "密史" in title or "野史" in title:
			priorgenre = infolist[2][i]
			if priorgenre == "小说":
				infolist[2][i] = "ny"
			elif priorgenre == "演义":
				infolist[2][i] = "yy"
			elif priorgenre == "志存记录":
				infolist[2][i] = "hy"

	'''
	if textbalance:
		if item_type == 0:
			dt = infolist[2]
		elif item_type == 1:
			dt = infolist[4]
		gs = Series(dt)

		vcgs = gs.value_counts()

		ungs = list(set(dt))
		genstart = []
		for ug in ungs:
			genstart.append(dt.index(ug))

		rangesets = []
		genstart= sorted(genstart)
		for i,it in enumerate(genstart):
			if i != (len(genstart)) -1:
				#print(i, it, genstart[i +1])
				randrange = [x for x in range(it,genstart[i+1])]
				#print(len(randrange))
				rangesets.append(randrange)
			else:
				#print(i,it)
				randrange = [x for x in range(it,len(dt))]
				#print(len(randrange))
				rangesets.append(randrange)

		reduced = []
		for rang in rangesets:

			red = random.sample(rang,vcgs[-1])
			reduced.extend(red)

		altinfo = []
		for i in range(0,len(infolist)):
			nl = []
			for it in reduced:
				nl.append(infolist[i][it])
			altinfo.append(nl)

		infolist = altinfo

	print("Making vectorizer")
	# create a vectorizer object to vectorize the documents into matrices. These
	# vectorizers return sparse matrices.

	# Calculate using plain term frequency
	if cnchoice == "tf":
		vectorizer = TfidfVectorizer(use_idf=False, analyzer='word', token_pattern='\S+', ngram_range=ngramrange, max_features=features, vocabulary=voc,norm='l2')

	# Calculate using TFIDF
	elif cnchoice == "tfidf":
		vectorizer = TfidfVectorizer(use_idf=True, analyzer='word', token_pattern='\S+', ngram_range=ngramrange, max_features=features,vocabulary=voc)

	# Calculate using raw term counts
	elif cnchoice == "raw":
		vectorizer = CountVectorizer(analyzer='word', token_pattern = '\S+', ngram_range=ngramrange, max_features=features,vocabulary=voc)

	# Calculate using a chi measure (based on Ted Underwood's tech note)
	# This returns a DataFrame and a list of vocabulary
	elif cnchoice == "chi":
		df, vocab = paulutility.chinormal(infolist[0], ngramrange, features, infolist[2])
		densematrix = df#.toarray()
	print("Fitting vectorizer")

	# create the Matrix if using a sklearn vectorizer object
	# this will finish with a matrix in the same form as the one returned
	# using the chi metric
	if cnchoice != "chi":
		matrix = vectorizer.fit_transform(infolist[0])
		vocab = vectorizer.get_feature_names()
		# A dense matrix is necessary for some purposes, so I convert the sparse
		# matrix to a dense one
		densematrix = matrix.toarray()
		if scalemeanzero:
			densematrix = scale(densematrix) #sklearn scale to mean 0, var 1

		df = DataFrame(densematrix, columns=vocab, index=infolist[2])




	################
	# PCA ANALYSIS #
	################

	#df = df[df[2] != "志存记录"]
	#print(df)
	if analysischoise == "pca":
		# run pca
		# by default I am only looking at the first two PCs
		pca = PCA(n_components=2)
		pca2 = PCA(n_components=2)
		pca2.fit(df)
		plt.figure(figsize=figsize)
		plt.plot(pca2.explained_variance_ratio_,marker='o')
		plt.xticks(np.arange(0,10,1))
		plt.xlabel('Principal Component')
		plt.ylabel('Explained Variance')
		plt.title('Scree Plot')
		plt.savefig(screeplotname)
		plt.clf()
		if item_type == 0:
			dt = infolist[2]
		elif item_type == 1:
			dt = infolist[4]
		seriesgenre = Series(dt)
		genrecount = seriesgenre.value_counts()
		print(genrecount)

		titleseries = Series(infolist[1])

		wf = open("usedtitles.txt","w")
		for title in set(infolist[1]):
			wf.write(title + "\n")
		wf.close()

		titlecount = titleseries.value_counts()
		print(titlecount)

		my_pca = pca.fit(df).transform(df) # same as PCA(n_components=2).fit_transform(df)

		# in sklearn, the loadings are held in pca.components_
		loadings = pca.components_

		# Pickle the loadings (useful for extra analysis), so
		# I don't have to reload data every time
		if pickle_loadings:
			pickle.dump([vocab,loadings], open('loadings.p','wb'))
		if plotloadings == True:
			# I first plot the loadings
			plt.figure(figsize=figsize)

			# Scatter plot using the loadings, needs work
			#plt.scatter(*loadings, alpha=0.0)
			plt.scatter(loadings[pcs[0]], loadings[pcs[1]], alpha=0.0)
			#plt.scatter([0,0],[0,0],alpha=0.0)

			# Label with explained variance
			pclabel1 = "PC"+str(pcs[0] + 1) + " "
			pclabel2 = "PC"+str(pcs[1] + 1) + " "
			plt.xlabel(pclabel1+str(pca.explained_variance_ratio_[pcs[0]]))
			plt.ylabel(pclabel2+str(pca.explained_variance_ratio_[pcs[1]]))

			# Set a Chinese Font. Mac compatible. Will need something else
			# on windows
			chinese = FontProperties(fname='/Library/Fonts/Songti.ttc')
			matplotlib.rc('font', family='STHeiti')

			# Iterate through the vocab and plot where it falls on loadings graph
			# numpy array the loadings info is held in is in the opposite format of the
			# pca information
			for i, txt in enumerate(vocab):
				plt.annotate(txt, (loadings[pcs[0], i], loadings[pcs[1], i]), horizontalalignment='center', verticalalignment='center', size=annottate_font_size)
			plt.title("Loadings" + fschoise + "  " + str(features))
			plt.savefig("loadings_" + savefigurename)
			# clear figure in prep for actual pca
			plt.clf()

		# Return information to use for making PCA graph
		# This returns information based on what I want
		# the labels to be. Unique classes is a list of n length
		# from zero to n of unique labels. y is a numpy array
		# with the label of each document transformed into a number from the
		# unique_classes list. Names is a list of the unique labels
		# color is a list of unique colors for each document, assigned
		# depending on the label. I later replace color with a hard coded
		# dictionary#
		unique_classes, y, names,color = paulutility.get_graph_info(infolist[label_name])

		plt.figure(figsize=figsize)

		# I am only working with four genres, so I hard coded a color dictionary
		# This makes it easier to work with.
		if item_type == 0:
			cdict = {"演义": "magenta", "正史":"blue", "小说": "green", "别史":"black", "野史":"gray", "ny":"gray", "yy":"purple", "剧曲":"purple", "文总集":"purple","宝卷":"purple","hy":"cyan", "志存记录":"black"}
		#markers = ['+', 'x', '*', '8', 'v']
		elif item_type == 1:
			cdict = {"王世贞":"blue", "徐渭":"green", "李开先":"magenta", "兰陵笑笑生":"black"}
		# Set Chinese font
		matplotlib.rc('font', family='STHeiti')

		# I used this dictionary to visibly plot just the yeshi
		#altalf = {"演义": 0, "正史":0, "小说": 0, "别史":0, "野史":1}

		# This plots each genre (or text, or directory) in a single color
		# code partially adapted from brandonrose.com
		# set color=color to use the color list returned by get_graph_info
		#h hardcode order


		for c, i, name in zip(color,unique_classes, names):

#			if i>=3:
			plt.scatter(my_pca[y==i, pcs[0]], my_pca[y==i, pcs[1]], label=name, color=cdict[name], marker='o', s=dot_size, alpha = alfa)

		# use explained variance as axis label
		pclabel1 = "PC"+str(pcs[0] + 1) + " "
		pclabel2 = "PC"+str(pcs[1] + 1) + " "
		plt.xlabel(pclabel1+str(pca.explained_variance_ratio_[pcs[0]]))
		plt.ylabel(pclabel2+str(pca.explained_variance_ratio_[pcs[1]]))

		# if you want a plot annotated with labels, This code will do that
#		wanttoannottate = "yes"
		if wanttoannottate == "yes":
			for i, txt in enumerate(infolist[1]):
#				if txt in ["野史", "志存记录"]:
				plt.annotate(txt, xy = (my_pca[i, pcs[0]], my_pca[i, pcs[1]]), xytext=(my_pca[i, pcs[0]], my_pca[i, pcs[1]]), size = annottate_font_size)

		plt.title(fschoise + "  " + str(features))
		plt.legend(loc=0)
		plt.savefig(savefigurename)
	################
	# HCA ANALYSIS #
	################
	elif analysischoise == "hca":
		matplotlib.rcParams['lines.linewidth'] = 0.5
		if item_type == 0:
			dt = infolist[2]
		elif item_type == 1:
			dt = infolist[4]
		seriesgenre = Series(dt)
		genrecount = seriesgenre.value_counts()
		print(genrecount)
		# run hca

		# Set similarity measure.
		print("creating similarity matrix")
		if distchoice == "cos":
			dis = cosine_similarity(densematrix)
		elif distchoice == "euc":
			dis = euclidean_distances(densematrix)

		print("creating linkage matrix")
		# make linkage matrix using ward algorithm
		cos_linkage_matrix = ward(dis)

		# get a dictionary that will create colors using a different thing than
		# what is used for the labels
		# coldict = paulutility.label_color_different(infolist[label_name], infolist[label_color])
		coldict = {"演义": "magenta", "正史":"blue", "小说": "green", "别史":"black", "野史":"black", "ny":"gray", "yy":"purple", "剧曲":"purple", "文总集":"purple","宝卷":"purple","hy":"cyan"}
		fig = plt.figure(figsize=(10,7))
		matplotlib.rc('font', family='STHeiti')
		plot1 = fig.add_subplot(1,1,1)
		# create dendrogram
		plot1 = dendrogram(cos_linkage_matrix, labels=infolist[label_name], color_threshold=20)
		# remove ticks on axes that done need them
		plt.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
		ax = plt.gca()
		# get labels
		labels = ax.get_xmajorticklabels()
		# reconfigure labels with new colors
		for lab in labels:
			#lab.set_rotation(90)
			lab.set_size(8)
			lab.set_color(coldict[lab.get_text()])
		#plt.legend(loc=0)
		newlabels = []
		for i in range(0,len(labels)):
			newlabels.append('|')
		ax.set_xticklabels(newlabels)


		plt.title(features)
		plt.xlabel("Texts")

		plt.savefig(savefigurename)