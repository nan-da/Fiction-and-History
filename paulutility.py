import os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
import random

DIVLIM = 20000

#  list of punctuation marks that need to be cleaned from most texts
puncs = ['』','。', '！', '，', '：', '、', '（', '）', '；', '？', '〉', '〈', '」', '「', '『', '“', '”', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '_', '`''{', '|', '}', '~', '¤', '±', '·', '×', 'à', 'á', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', '÷', 'ù', 'ú', 'ü', 'ā', 'ī', 'ń', 'ň', 'ō', 'ū', 'ǎ', 'ǐ', 'ǔ', 'ǖ', 'ǘ', 'ǚ', 'ǜ', 'ǹ', 'ɑ', 'ɡ', 'α', 'β', 'γ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'а', 'б', 'в', 'г', 'д', 'е', 'к', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', '—', '‖', '‘', '’', '…', '※', 'ⅰ', 'ⅲ', '∈', '∏', '∑', '√', '∠', '∥', '∧', '∩', '∪', '∫', '∮', '∶', '∷', '∽', '≈', '≌', '≡', '⊙', '⊥', '⌒', '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '⑴', '⑵', '⑶', '⑷', '⑸', '⑹', '⑺', '⑻', '⑼', '⑽', '⑾', '⑿', '⒀', '⒁', '⒂', '⒃', '⒄', '⒅', '⒆', '⒈', '⒉', '⒊', '⒋', '⒌', '⒍', '⒎', '⒏', '⒐', '⒑', '⒒', '⒓', '⒔', '⒕', '⒖', '⒗', '⒘', '⒙', '⒚', '⒛', '─', '┅', '┋', '┌', '┍', '┎', '┏', '┐', '┑', '┒', '┓', '└', '┕', '┘', '┙', '┚', '┛', '├', '┝', '┞', '┠', '┡', '┢', '┣', '┤', '┥', '┦', '┧', '┩', '┪', '┫', '┬', '┭', '┮', '┯', '┰', '┱', '┲', '┳', '■', '□', '▲', '△', '◆', '◇', '○', '◎', '●', '★','︶', '﹑', '﹔', '﹖', '＂', '＃', '％', '＆', '＊','．', '／', '０', '１', '２', '３', '４', '５', '６', '７', '８', '９', '＜', '＝', '＞', '＠', '［', '＼', '］', '＿', '｀', 'ａ', 'ｂ', 'ｃ', 'ｄ', 'ｅ', 'ｆ', 'ｇ', 'ｈ', 'ｉ', 'ｊ', 'ｋ', 'ｌ', 'ｍ', 'ｎ', 'ｏ', 'ｐ', 'ｑ', 'ｒ', 'ｓ', 'ｔ', 'ｕ', 'ｖ', 'ｗ', 'ｘ', 'ｙ', 'ｚ', '｛', '｝', '～', '￥','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','《', '》', '〔', '〕', '【', '】', 'A',  'B',  'C',  'D',  'E',  'F',  'G',  'H', 'I', 'J', 'K', 'L', "M", 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',  'Ｗ',  'Ｘ',  'Ｙ',  'Ｚ',  '＾',  '｜', '￠',  '￡']

# function to open and read a file. returns a string
def getcontent(filename):
	f = open(filename,'r')
	content = f.read()
	f.close()
	return content

# reduces the length of the filename. Some files are labeled with
# 06-01-001. Others are 06-01-002-01. This gets rid of the last 01
def namereduce(filename):
	fsplit = filename.split('-')
	if len(fsplit) == 4:
		filename = '-'.join(fsplit[:-1])
	return filename

# adds 0 or 00 to labels when there are more than 10 or 100 total files
def addlabel(i, totallen):
	if totallen > 99:
		if i >= 10 and i < 99:
			num = "0" + str(i)
		elif i < 10:
			num = "00" + str(i)
		else:
			num = str(i)
	elif totallen > 10:
		if i < 10:
			num = "0" + str(i)
		else:
			num = str(i)
	else:
		num = str(i)
	return num

# reads metadata file found in the directory
# this file contains a line for each file in the form of
# filename, genre, text title, dynasty,  author
# returns a dictionary using the filename as key, list for the rest of the info
def readmeta(filename):
	content = getcontent(filename)
	dictionary = {}
	lines = content.split('\n')
	for line in lines:
		cells = line.split('\t')
		fname = cells[0]
		fname = namereduce(fname)
		genre = cells[1]
		title = cells[2]
		era = cells[3]
		author = cells[4]
		dictionary[fname] = [title, genre, era, author]
	return dictionary

# clean the text
def clean(content):
	content = re.sub('\s+', '', content)
	for punc in puncs:
		content = content.replace(punc, "")

	return " ".join(list(content))

# clean the text, remove the division flag
def cleanplus(content):
	content = re.sub('\s+', '', content)
	content = re.sub('~~~START\|.+?\|START~~~','', content)
	for punc in puncs:
		content = content.replace(punc, "")

	return " ".join(list(content))

# split text into evenly divided parts.
# takes a string as input. the default division is set at the top of this file
# left over portions of text that don't meat the minimum length are dumped by default
def numsplit(inputtext, divlim = DIVLIM, dumpsmall=False):

	# clean the text, returns a space delimited string
	text = clean(inputtext)

	# split into character tokens
	tokens = text.split(" ")

	# Calculate number of loops to run
	loops = len(tokens)//divlim + 1

	# Empty list for indivdual sections
	save = []

	# loop to seperate sections
	for i in range(0, loops):
		save.append(" ".join(tokens[i * divlim: (i + 1) * divlim]))

	# if dumping small, check length of last item in list. if less than
	# div number, dump it
	if dumpsmall:
		if len(save[-1]) < ((DIVLIM * 2) -1):
			save = save[:-1]
	# else, only save it if it is at least 1000 characters long (2000 because of spaces)
	# this is arbitrary, but this is on the small end of the texts I work with
	else:
		if len(save[-1]) < 2000:
			save = save[:-1]
	# returns the list with cleaned tokenized and normalized texts
	return save

# same as above but removes internal division markers as well
def numsplit2(inputtext, divlim = DIVLIM, dumpsmall=False):
	text = cleanplus(inputtext)
	tokens = text.split(" ")
	save = []

	loops = len(tokens)//divlim + 1
	for i in range(0, loops):
		save.append(" ".join(tokens[i * divlim: (i + 1) * divlim]))
	if dumpsmall:
		if len(save[-1]) < ((DIVLIM * 2) -1):
			save = save[:-1]
	else:
		if len(save[-1]) < 2000:
			save = save[:-1]
	return save

# Split each text into a document with a defined number of characters
def fullsplitbynum(directories, divs, kg, dt, dump=True, e=None):
	# set up save lists
	contents, titles, genres, eras = [],[],[],[]
	authors, dir, sectitles, fnames = [],[],[],[]
	lexdivs, doclengths = [], []
	# Iterate through each directory
	for d in directories:

		# read metadata file, get a dictionary
		metadictionary = readmeta("metadata.txt")

		# open each document in the directory
		for root, dirs, files in os.walk(d):
			if '.DS_Store' in files:
				files.remove('.DS_Store')
			for f in files:
				# reduce the filename
				filename = namereduce(f)
				if ".txt" in filename:
					filename = filename[:-4]

				# return the list that pertains to file
				if filename not in metadictionary:
					continue #because some English title files are not in metadata.txt
				metainfo = metadictionary[filename]


				# Check if the genre is to be included in analysis
				# if so, continue
				if dt == 0:
					usemeta = metainfo[1]
				elif dt == 1:
					usemeta = metainfo[3]
				if usemeta in kg:
					if e == None:
						# open the document
						c = getcontent(root+"/"+f)

						# split the text
						numtexts = numsplit2(c, divlim = divs, dumpsmall=dump)

						# iterate through each section of the text
						for i, txt in enumerate(numtexts):
							# if the text is of suitable length
							# continue analysis. This checks to make sure the returned
							# document is at least as long as the division length
							if len(txt) >= (divs * 2) - 1:
								# add a label
								label =  addlabel(i, len(numtexts))
								# append various metainfos to lists
								titles.append(metainfo[0])
								genres.append(metainfo[1])
								eras.append(metainfo[2])
								authors.append(metainfo[3])
								dir.append(d)
								# append contents to list
								contents.append(txt)
								sectitles.append(label)
								fnames.append(f)

								chars = txt.split(" ")
								lexdivs.append(len(set(chars))/len(chars))
								doclengths.append(len(chars))
					elif e != None:
						if metainfo[2] in e:
							c = getcontent(root+"/"+f)

							# split the text
							numtexts = numsplit2(c, divlim = divs, dumpsmall=dump)

							# iterate through each section of the text
							for i, txt in enumerate(numtexts):
								# if the text is of suitable length
								# continue analysis. This checks to make sure the returned
								# document is at least as long as the division length
								if len(txt) >= (divs * 2) - 1:
									# add a label
									label =  addlabel(i, len(numtexts))
									# append various metainfos to lists
									titles.append(metainfo[0])
									genres.append(metainfo[1])
									eras.append(metainfo[2])
									authors.append(metainfo[3])
									dir.append(d)
									# append contents to list
									contents.append(txt)
									sectitles.append(label)
									fnames.append(f)
									chars = txt.split(" ")
									lexdivs.append(len(set(chars))/len(chars))
									doclengths.append(len(chars))
	# return contents and metainformation for use in main program
	return [contents, titles, genres, eras, authors, dir, sectitles, fnames, lexdivs, doclengths]

# Get full text information. Same as above, but does not divide the text
def fulltextcontent(directories, kg, dt, e=None): #dt = item_type = 0 for genre, 1 for author
	contents, titles, genres, eras = [],[],[],[]
	authors, dir, sectitles, fnames = [],[],[],[]
	lexdivs, doclengths = [], []
	for d in directories:
		metadictionary = readmeta("metadata.txt")
		for root, dirs, files in os.walk(d):
			if '.DS_Store' in files:
				files.remove('.DS_Store')
			for f in files:
				filename = namereduce(f)
				if ".txt" in filename:
					filename = filename[:-4]
				metainfo = metadictionary[filename]
				if dt == 0:
					usemeta = metainfo[1]
				elif dt == 1:
					usemeta = metainfo[3]
				if usemeta in kg:
					if e == None:
						titles.append(metainfo[0])
						genres.append(metainfo[1])
						eras.append(metainfo[2])
						authors.append(metainfo[3])
						dir.append(d)
						c = getcontent(root+"/"+f)
						c = cleanplus(c) # remove punctuations, and join characters with a space
						contents.append(c)
						fnames.append(f)
						chars = c.split(" ")
						lexdivs.append(len(set(chars))/len(chars))
						doclengths.append(len(chars))
					elif e != None:
						if metainfo[2] in e:
							titles.append(metainfo[0])
							genres.append(metainfo[1])
							eras.append(metainfo[2])
							authors.append(metainfo[3])
							dir.append(d)
							c = getcontent(root+"/"+f)
							c = cleanplus(c)
							contents.append(c)
							fnames.append(f)
							chars = c.split(" ")
							lexdivs.append(len(set(chars))/len(chars))
							doclengths.append(len(chars))

	return [contents, titles, genres, eras, authors, dir, sectitles, fnames, lexdivs, doclengths]

# Get split text information. Same as above, but divides using the internal
# divisions delineated with ~~~START|sectiontitle|START~~~
# If a section is longer than 50,000 characters, it will split it
def splittextcontent(directories, kg, dt, e=None):
	contents = []
	titles = []
	sectitles = []
	genres = []
	eras = []
	authors = []
	dir = []
	fnames = []
	lexdivs = []
	doclengths = []
	for d in directories:

		metadictionary = readmeta("metadata.txt")
		for root, dirs, files in os.walk(d):
			if '.DS_Store' in files:
				files.remove('.DS_Store')
			for f in files:
				filename = namereduce(f)
				if ".txt" in filename:
					filename = filename[:-4]
				metainfo = metadictionary[filename]
				if dt == 0:
					usemeta = metainfo[1]
				elif dt == 1:
					usemeta = metainfo[3]

				if usemeta in kg:

					if e == None:
						c = getcontent(root+"/"+f)
						sections = re.split('~~~START\|(.+?)\|START~~~', c)
						sections = sections[1:]
						#print(len(sections))

						for i, sec in enumerate(sections):
							if i%2 == 0:
								text = sections[i+1]
								label = sec
								if len(text) > 50000:
									divided = numsplit(text)
									for div in divided:
										contents.append(text)
										sectitles.append(label)
										titles.append(metainfo[0])
										genres.append(metainfo[1])
										eras.append(metainfo[2])
										authors.append(metainfo[3])
										dir.append(d)
										fnames.append(f)
										chars = txt.split(" ")
										lexdivs.append(len(set(chars))/len(chars))
										doclengths.append(len(chars))
								elif len(text) >= 1000:
									if ".com" in text:
										print(f)
									text = clean(text)
									contents.append(text)
									sectitles.append(label)
									titles.append(metainfo[0])
									genres.append(metainfo[1])
									eras.append(metainfo[2])
									authors.append(metainfo[3])
									dir.append(d)
									fnames.append(f)
									chars = txt.split(" ")
									lexdivs.append(len(set(chars))/len(chars))
									doclengths.append(len(chars))
					elif e != None:
						if metainfo[2] in e:
							c = getcontent(root+"/"+f)
							sections = re.split('~~~START\|(.+?)\|START~~~', c)
							sections = sections[1:]
							#print(len(sections))

							for i, sec in enumerate(sections):
								if i%2 == 0:
									text = sections[i+1]
									label = sec
									if len(text) > 50000:
										divided = numsplit(text)
										for div in divided:
											contents.append(text)
											sectitles.append(label)
											titles.append(metainfo[0])
											genres.append(metainfo[1])
											eras.append(metainfo[2])
											authors.append(metainfo[3])
											dir.append(d)
											fnames.append(f)
											chars = txt.split(" ")
											lexdivs.append(len(set(chars))/len(chars))
											doclengths.append(len(chars))
									elif len(text) >= 1000:
										if ".com" in text:
											print(f)
										text = clean(text)
										contents.append(text)
										sectitles.append(label)
										titles.append(metainfo[0])
										genres.append(metainfo[1])
										eras.append(metainfo[2])
										authors.append(metainfo[3])
										dir.append(d)
										fnames.append(f)
										chars = txt.split(" ")
										lexdivs.append(len(set(chars))/len(chars))
										doclengths.append(len(chars))
	return [contents, titles, genres, eras, authors, dir, sectitles, fnames, lexdivs, doclengths]


# Gets n colors for later functions
# This code comes from stack overflow. It returns an n length list of unique colors
def get_ncolors(N):
	color_norm = colors.Normalize(vmin=0,vmax=N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
	cs = []
	for i in range(0,N):
		cs.append(scalar_map.to_rgba(i))
	return cs

# Gets info for stylometry, returns a mapping of unique class names to integers,
# a numpy array of integers as class labels, the unique class names, and an a list
# of colors
def get_graph_info(list_of_items):
	uniqueitems = set(list_of_items)
	uniqueitems=['正史', '小说', '演义', '志存记录','野史']
	print(uniqueitems)
	classes = [i for i in range(0, len(uniqueitems))]
	udict = dict(zip(uniqueitems,classes))
	classlist=[]
	for item in list_of_items:
		classlist.append(udict[item])
	col = get_ncolors(len(classes))
	return classes, np.array(classlist), uniqueitems, col

# This function returns a color dictionary. This also the color of labels
# in HCA to differ from the label names
def label_color_different(labels, thingforcolor):
    cols = get_ncolors(len(set(thingforcolor)))
    coldict = {}
    cols = ["Red", "Green", "Magenta", "Blue"]
    for thing, col in zip(set(thingforcolor), cols):
        coldict[thing] = col
    labtocol = {}
    #coldict = {"演义": "magenta", "正史":"blue", "小说": "green", "别史":"black", "野史":"black", "诗话":"gray"}
    for label, thing in zip(labels, thingforcolor):
        labtocol[label] = coldict[thing]
    print(coldict)
    return labtocol

# implements Ted Underwood's scoring system. Essentially finds chi
# takes a list of texts, an ngram range, a set number of features,
# and defines what to use as the dataframe index
# returns the weighted dataframe and the feature vocab
# The ngram range is meant to duplicate functionality offered by
# the sklearn vectorizers
def chinormal(textstoparse, ngramrange, features, indexinfo):
	alltoks = []
	for ind, t in enumerate(textstoparse):
		tokenized = t.split(" ")
		if ngramrange == (1,1):
			toks = Series(tokenized, name=indexinfo[ind])
		else:
			allgrams = []
			mingram = ngramrange[0]
			maxgram = ngramrange[1]
			for j in range(mingram, maxgram + 1):
				for i in range(0, len(tokenized)-j + 1):
					tok = "".join(tokenized[i:i+j])
					allgrams.append(tok)
			toks = Series(allgrams, name=indexinfo[ind])

		tc = toks.value_counts()
		alltoks.append(tc)

	df = pd.concat(alltoks, axis=1)
	df = df.T
	df = df.fillna(0)
	chartotal = df.sum()
	charorder = chartotal.sort_values(ascending=False)
	featvocab = charorder[:features].index
	doclen = df.sum(axis=1)
	corpussize = doclen.sum()
	charbycorp = (chartotal/corpussize)
	df2 = DataFrame(1, index=doclen.index, columns=charbycorp.index)
	df2 = df2.multiply(doclen,axis='index') * charbycorp
	res = df[featvocab] - df2[featvocab]
	return res, featvocab
