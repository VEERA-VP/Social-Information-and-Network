twitter_storing.py
from tweepy import 
StreamListener from tweepy 
import OAuthHandler from 
tweepy import Stream
from tweepy import API 
from tweepy import 
Cursor
#Custom stream listener class to collect relevant 
details class StreamListener(StreamListener):
def on_status(self, status):
if hasattr(status, 'retweeted_status'): # if the tweet is a retweeted tweet it 
return #will give the control back to
stream.fliter
print(status.text+"\n") 
fp.write(status.text+"\n")
def on_error(self, status_code):
if status_code == 420:
return False
#API keys
ckey = '#######################HJYjVs2sD9#########'
csec = 
'#######################VvQBnLyEl7g#############
#' atok = 
'#######################ocuZF7sLUpDBflKip#######' 
asec = '#######################35qGWNRKdZ3J###'
#Auth
auth = OAuthHandler(ckey, 
csec) 
auth.set_access_token(atok, 
asec) api = API(auth)

# \U0001F631 refers to 'confused'
#1F602 = happy, 1F62D = sad, 1F621= angry, 2764 = love, 1F61C = playful, 1F631 
= confused query = [u'\U0001F602']
CSE3021–SOCIAL AND INFORMATIONNETWORKS–J COMPONENTPROJECTWORK REPORT
35 | P a g 
e
fp = open("store.txt",'w',encoding='utf-8')
stream = Stream(auth = api.auth, listener = StreamListener()) 
stream.filter(track = query, languages = ["en"], stall_warnings = True) 
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@")
process.py
import re
fp = open("store.txt",'r',encoding='utf-8')
processedFile = 
open("processed.txt",'w',encoding='utf-8') emojiFile = 
open("emojiFile.txt",'w',encoding='utf-8')
for line in fp:
line = " ".join(filter(lambda x:x[0]!='@', line.split()))
line = " ".join(filter(lambda x:x[0:8]!="https://", line.split()))
#line = " ".join(filter(lambda x:x, 
line.split(','))) if len(line)<2:
continue 
processedFile.write(line+"\n") 
print(line)
emojis = re.findall(r'[^\w\s,. ]', 
line) emojis = " ".join(emojis)
'''lst =[]
if (len(emojis)<2):
continue
for emoji in emojis:
if emoji in ['#','@','!',':','\'','\"','\\',",",'?','<','>','.',';','-','&','/']:
continu
e 
lst.append(emoj
i)
lst = " ".join(lst) 
emojiFile.write(lst+"\n")''' 
emojiFile.write(emojis+"\n")
fp.close() 
processedFile.close() 
emojiFile.close()
SentimentAnalyser.py
import os 
import math 
import string 
import 
codecs 
import json
from io import 
open import 
numpy as np
from itertools import product 
from inspect import 
getsourcefile
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# sentiments intensity rating increase/decrease for booster 
words. bIncr = 0.293
bDecr = -0.293
# sentiment intensity increases for ALL Caps 
words cIncr = 0.733
nScalar = -0.74
# 
http://en.wiktionary.org/wiki/Category:English_degree_adver
bs negate = \
["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
"ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
"dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
"don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
"neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere", 
"oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
"oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
"without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
boosterDict = \
{"absolutely": bIncr, "amazingly": bIncr, "awfully": bIncr, 
"completely": bIncr, "considerable": bIncr, "considerably": 
bIncr,
"decidedly": bIncr, "deeply": bIncr, "effing": bIncr, "enormous": bIncr, "enormously": 
bIncr, "entirely": bIncr, "especially": bIncr, "exceptional": bIncr, "exceptionally": 
bIncr, "extreme": bIncr, "extremely": bIncr,
"fabulously": bIncr, "flipping": bIncr, "flippin": bIncr, "frackin": bIncr, "fracking": 
bIncr, "fricking": bIncr, "frickin": bIncr, "frigging": bIncr, "friggin": bIncr, "fully": 
bIncr, "fuckin": bIncr, "fucking": bIncr, "fuggin": bIncr, "fugging": bIncr,
"greatly": bIncr, "hella": bIncr, "highly": bIncr, "hugely": 
bIncr, "incredible": bIncr, "incredibly": bIncr, "intensely": 
bIncr,
"major": bIncr, "majorly": bIncr, "more": bIncr, "most": bIncr, "particularly": bIncr, 
"purely": bIncr, "quite": bIncr, "really": bIncr, "remarkably": bIncr,
"so": bIncr, "substantially": bIncr,
"thoroughly": bIncr, "total": bIncr, "totally": bIncr, "tremendous": bIncr, "tremendously": 
bIncr, "uber": bIncr, "unbelievably": bIncr, "unusually": bIncr, "utter": bIncr, "utterly": bIncr,
"very": bIncr,
"almost": bDecr, "barely": bDecr, "hardly": bDecr, "just enough": 
bDecr, "kind of": bDecr, "kinda": bDecr, "kindof": bDecr, "kind-of": 
bDecr, "less": bDecr, "little": bDecr, "marginal": bDecr, 
"marginally": bDecr, "occasional": bDecr, "occasionally": bDecr, 
"partly": bDecr,
"scarce": bDecr, "scarcely": bDecr, "slight": bDecr, "slightly": bDecr, "somewhat": 
bDecr, "sort of": bDecr, "sorta": bDecr, "sortof": bDecr, "sort-of": bDecr}
# some commonly used idoms
sentimentLadenIdioms = {"cut the mustard": 2, "hand to mouth": -2, 
"back handed": -2, "blow smoke": -2, "blowing 
smoke": -2, "upper hand": 1, "break a leg": 2,
"cooking with gas": 2, "in the black": 2, "in the red": 
-2, "on the ball": 2, "under the weather": -2}
# check for special case idioms and phrases containing lexicon words
specialCases = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "badass": 1.5, "bus 
stop": 0.0, "yeah right": -2, "kiss of death": -1.5, "to die for": 3,
"beating heart": 3.1, "broken heart": -2.9 }

def negated(inputWords, includeNt=True):
"""
Determine if input contains negation 
words """
inputWords = [str(w).lower() for w in 
inputWords] negWords = []
negWords.extend(negate) 
for word in negWords:
if word in inputWords:
return True
if includeNt: # shouldn't, couldn't; all the n't 
words for word in inputWords:
if "n't" in word:
return 
True return 
False
def normalize(score, alpha=15):
"""
Normalize the score to be between -1 and 1 using an 
alpha that approximates the max expected value
"""
normScore = score / math.sqrt((score * score) + 
alpha) if normScore < -1.0:
return -1.0
elif normScore > 1.0:
return 
1.0 else:
return normScore
def allCapitalDifferent(words):
"""
Check whether just some words in the input are ALL CAPS
:returns: `True` if some but not all items in `words` are ALL 
CAPS """
allCapitalWords = 
0 for word in 
words:
if word.isupper():
allCapitalWords += 1
cap_differential = len(words) -
allCapitalWords if 0 < cap_differential < 
len(words):
return 
True else:
return False
def scalarIncDec(word, valence, isCapitalDiff):
"""
Check if the preceding words increase, decrease, or 
negate/nullify the valence
"""
scalar = 0.0

wordLower = 
word.lower() if wordLower 
in boosterDict:
scalar =
boosterDict[wordLower] if 
valence < 0:
scalar *= -1
if word.isupper() and isCapitalDiff:
if valence > 0:
scalar += 
cIncr else:
scalar -= 
cIncr return scalar
class SentiText(object):
"""
Identify sentiment-relevant string-level properties of input 
text. """
def init (self, text):
if not isinstance(text, str):
text = str(text).encode('utf8') self.text = text
self.wordsAndEmoticons = self._wordsAndEmoticons()
self.isCapitalDiff = allCapitalDifferent(self.wordsAndEmoticons) #Check whether some words are CAPS
@staticmethod
def _StripPuncInWord(token):
"""
Removes all trailing and leading punctuation
If the resulting string has two or fewer characters, 
then it was likely an emoticon, so return original 
string (ie ":)" stripped would be "", so just return 
":)"
"""
stripped = 
token.strip(string.punctuation) if 
len(stripped) <= 2:
return token 
return 
stripped
def _wordsAndEmoticons(self):
"""
Removes leading and trailing 
puncutation Leaves contractions and 
most emoticons
Does not preserve punc-plus-letter emoticons 
(e.g. :D) """
wes = self.text.split()
stripped = list(map(self._StripPuncInWord, 
wes)) return stripped
class SentimentIntensityAnalyzer(object):
"""
Give a sentiment intensity score to
sentences. """
def init (self, lexiconFile="vader_lexicon.txt", emojiLexicon="emoji_utf8_lexicon.txt"):

_thisModuleFilePath_ = os.path.abspath(getsourcefile(lambda: 0))
lexiconFullFilePath = os.path.join(os.path.dirname(_thisModuleFilePath_), lexiconFile)

with codecs.open(lexiconFullFilePath, encoding='utf-8') 
as f: self.lexiconFullFilePath = f.read()
self.lexicon = self.makeLexiconDictionary()
emojiFilePath = os.path.join(os.path.dirname(_thisModuleFilePath_), 
emojiLexicon) with codecs.open(emojiFilePath, encoding='utf-8') as f:
self.emojiFilePath = f.read() 
self.emojis =
self.makeEmojisDictionary()
def makeLexiconDictionary(self):
"""
Convert lexicon file to a 
dictionary """
lexiconDict = {}
for line in 
self.lexiconFullFilePath.rstrip('\n').split('\n'): if 
not line:
continue
(word, measure) = 
line.strip().split('\t')[0:2]
lexiconDict[word] = float(measure)
return lexiconDict
def makeEmojisDictionary(self):
"""
Convert emoji lexicon file to a
dictionary """
EmojisDict = {}
for line in 
self.emojiFilePath.rstrip('\n').split('\n'): 
(emoji, description) = 
line.strip().split('\t')[0:2]
EmojisDict[emoji] = 
description return EmojisDict
def polarityScore(self, text):
"""
Return a float for sentiment strength based on the input 
text. Positive values are positive valence, negative value 
are negative valence.
"""
textAndEmoji = 
"" prevSpace = 
True for chr in 
text:
if chr in self.emojis:
description =
self.emojis[chr] if not
prevSpace:
textAndEmoji += ' ' 
textAndEmoji += 
description prevSpace = 
False
else:
textAndEmoji += chr 
prevSpace = chr == ' '
text = 

textAndEmoji.strip() 
sentitext =
SentiText(text)
sentiments = []
wordsAndEmoticons = sentitext.wordsAndEmoticons

for i, item in enumerate(wordsAndEmoticons):
valence = 0
if item.lower() in boosterDict:
sentiments.append(valence
) continue
if (i < len(wordsAndEmoticons) - 1 and item.lower() == 
"kind" and wordsAndEmoticons[i + 1].lower() == "of"):
sentiments.append(valence
) continue
sentiments = self.sentimentValence(valence, sentitext, item, i, 
sentiments) sentiments = self._checkBut(wordsAndEmoticons, 
sentiments)
valenceDict = self.valenceScore(sentiments, 
text) return valenceDict
def sentimentValence(self, valence, sentitext, item, i, sentiments):
isCapitalDiff = sentitext.isCapitalDiff 
wordsAndEmoticons = 
sentitext.wordsAndEmoticons itemLowerCase = 
item.lower()
if itemLowerCase in self.lexicon:
valence = self.lexicon[itemLowerCase]
if itemLowerCase == "no" and i != len(wordsAndEmoticons)-1 and wordsAndEmoticons[i + 
1].lower() in self.lexicon:
valence = 0.0
if (i > 0 and wordsAndEmoticons[i - 1].lower() == 
"no") \ or (i > 1 and wordsAndEmoticons[i -
2].lower() == "no")\
or (i > 2 and wordsAndEmoticons[i - 3].lower() == "no" and wordsAndEmoticons[i - 1].lower() in ["or",
"nor"] ):
valence = self.lexicon[itemLowerCase] * nScalar
if item.isupper() and isCapitalDiff:
if valence > 0:
valence += 
cIncr else:
valence -= cIncr 
for startI in range(0,
3):
if i > startI and wordsAndEmoticons[i - (startI + 1)].lower() not in 
self.lexicon: s = scalarIncDec(wordsAndEmoticons[i - (startI + 1)], 
valence, isCapitalDiff) if startI == 1 and s != 0:
s = s * 0.95
if startI == 2 and s != 0:
s = s * 0.9

valence = valence + s
valence = self._negationCheck(valence, wordsAndEmoticons, startI, i)

if startI == 2:
valence = self._specialIdiomsCheck(valence, wordsAndEmoticons, i)
valence = self._checkLeast(valence, 
wordsAndEmoticons, i) sentiments.append(valence)
return sentiments
def _checkLeast(self, valence, wordsAndEmoticons, i):
if i > 1 and wordsAndEmoticons[i - 1].lower() not in 
self.lexicon \ and wordsAndEmoticons[i - 1].lower() == 
"least":
if wordsAndEmoticons[i - 2].lower() != "at" and wordsAndEmoticons[i - 2].lower() != 
"very": valence = valence * nScalar
elif i > 0 and wordsAndEmoticons[i - 1].lower() not in 
self.lexicon \ and wordsAndEmoticons[i - 1].lower() == 
"least":
valence = valence * 
nScalar return valence
@staticmethod
def _checkBut(wordsAndEmoticons, sentiments):
wordsAndEmoticonsLower = [str(w).lower() for w in 
wordsAndEmoticons] if 'but' in wordsAndEmoticonsLower:
bi = 
wordsAndEmoticonsLower.index('but') 
for sentiment in sentiments:
si = 
sentiments.index(sentiment) if 
si < bi:
sentiments.pop(si) 
sentiments.insert(si, sentiment * 
0.5)
elif si > bi:
sentiments.pop(si) 
sentiments.insert(si, sentiment * 
1.5)
return sentiments
@staticmethod
def _specialIdiomsCheck(valence, wordsAndEmoticons, i): 
wordsAndEmoticonsLower = [str(w).lower() for w in 
wordsAndEmoticons]
onezero = "{0} {1}".format(wordsAndEmoticonsLower[i - 1], wordsAndEmoticonsLower[i])
twoonezero = "{0} {1} {2}".format(wordsAndEmoticonsLower[i - 2], 
wordsAndEmoticonsLower[i - 1], 
wordsAndEmoticonsLower[i])
twoone = "{0} {1}".format(wordsAndEmoticonsLower[i - 2], wordsAndEmoticonsLower[i - 1])
threetwoone = "{0} {1} {2}".format(wordsAndEmoticonsLower[i - 3],
wordsAndEmoticonsLower[i - 2], wordsAndEmoticonsLower[i - 1])
threetwo = "{0} {1}".format(wordsAndEmoticonsLower[i - 3], 
wordsAndEmoticonsLower[i - 2]) sequences = [onezero, twoonezero, twoone, 
threetwoone, threetwo]

for seq in sequences:
if seq in specialCases:
valence = 
specialCases[seq] break
if len(wordsAndEmoticonsLower) - 1 > i:

zeroone = "{0} {1}".format(wordsAndEmoticonsLower[i], 
wordsAndEmoticonsLower[i + 1]) if zeroone in specialCases:
valence = specialCases[zeroone]
if len(wordsAndEmoticonsLower) - 1 > i + 1:
zeroonetwo = "{0} {1} {2}".format(wordsAndEmoticonsLower[i], wordsAndEmoticonsLower[i + 1],
wordsAndEmoticonsLower[i + 
2]) if zeroonetwo in specialCases:
valence = specialCases[zeroonetwo]
nGrams = [threetwoone, threetwo, 
twoone] for nGram in nGrams:
if nGram in boosterDict:
valence = valence + 
boosterDict[nGram] return valence
@staticmethod
def _sentimentLadenIdiomsCheck(valence, sentimentTextLower):
idiomsValences = []
for idiom in sentimentLadenIdioms:
if idiom in sentimentTextLower:
print(idiom, sentimentTextLower) 
valence = 
sentimentLadenIdioms[idiom] 
idiomsValences.append(valence)
if len(idiomsValences) > 0:
valence = sum(idiomsValences) / 
float(len(idiomsValences)) return valence
@staticmethod
def _negationCheck(valence, wordsAndEmoticons, startI, i): 
wordsAndEmoticonsLower = [str(w).lower() for w in 
wordsAndEmoticons] if startI == 0:
if negated([wordsAndEmoticonsLower[i - (startI + 1)]]):
valence = valence * 
nScalar if startI == 1:
if wordsAndEmoticonsLower[i - 2] == "never" 
and \ (wordsAndEmoticonsLower[i - 1] == 
"so" or wordsAndEmoticonsLower[i - 1] == 
"this"):
valence = valence * 1.25
elif wordsAndEmoticonsLower[i - 2] == "without" 
and \ wordsAndEmoticonsLower[i - 1] == 
"doubt":
valence = valence
elif negated([wordsAndEmoticonsLower[i - (startI + 1)]]):
valence = valence * 
nScalar if startI == 2:
if wordsAndEmoticonsLower[i - 3] == "never" and \
(wordsAndEmoticonsLower[i - 2] == "so" or wordsAndEmoticonsLower[i - 2] == 
"this") or \ (wordsAndEmoticonsLower[i - 1] == "so" or wordsAndEmoticonsLower[i 
- 1] == "this"):
valence = valence * 1.25
elif wordsAndEmoticonsLower[i - 3] == "without" and \
(wordsAndEmoticonsLower[i - 2] == "doubt" or wordsAndEmoticonsLower[i - 1] == 
"doubt"): valence = valence
elif negated([wordsAndEmoticonsLower[i - (startI + 
1)]]): valence = valence * nScalar
return valence

def _punctutaionEmphasis(self, text):
epAmp = self._ampEp(text) 
qmAmp = self._ampQm(text) 
puctEmphAmp = epAmp + 
qmAmp return puctEmphAmp
@staticmethod 
def 
_ampEp(text):
epCount =
text.count("!") if 
epCount > 4:
epCount = 4
epAmp = epCount * 
0.292 return epAmp
@staticmethod 
def
_ampQm(text):
qmCount = 
text.count("?") qmAmp = 
0
if qmCount > 1:
if qmCount <= 3:
qmAmp = qmCount * 
0.18 else:
qmAmp = 
0.96 return 
qmAmp
@staticmethod
def _shiftSentimentsScore(sentiments):
positiveSum = 0.0
negativeSum = 0.0
neutralCount = 0
for sentimentScore in sentiments:
if sentimentScore > 0:
positiveSum += (float(sentimentScore) 
+ 1) if sentimentScore < 0:
negativeSum += (float(sentimentScore) 
- 1) if sentimentScore == 0:
neutralCount += 1
return positiveSum, negativeSum, neutralCount
def valenceScore(self, sentiments, 
text): if sentiments:
sumSenti = float(sum(sentiments))
puctEmphAmp = 
self._punctutaionEmphasis(text) if sumSenti 
> 0:
sumSenti += 

puctEmphAmp elif 
sumSenti < 0:
sumSenti -= puctEmphAmp

compound = normalize(sumSenti)
positiveSum, negativeSum, neutralCount = 
self._shiftSentimentsScore(sentiments) if positiveSum > 
math.fabs(negativeSum):
positiveSum += puctEmphAmp
elif positiveSum < math.fabs(negativeSum):
negativeSum -= puctEmphAmp
total = positiveSum + math.fabs(negativeSum) + 
neutralCount pos = math.fabs(positiveSum / total)
neg = math.fabs(negativeSum / 
total) neu = 
math.fabs(neutralCount / total)
else:
compound = 0.0
pos = 0.0
neg = 0.0
neu = 0.0
sentiment_dict = \
{"neg": round(neg, 3),
"neu": round(neu, 3),
"pos": round(pos, 3),
"compound": round(compound, 
4)} return sentiment_dict
def visualize(compoundScore): 
compoundPositive,compoundNegative, compoundNeutral
=0,0,0 for i in compoundScore:
if i>0:
compoundPositive+=
1 elif i<0:
compoundNegative+= 
1 else:
compoundNeutral+=1
print("Number of Negative Emotions detected:-
"+str(compoundNegative)) print("Number of Neutral Emotions 
detected:- "+str(compoundNeutral)) print("Number of Postive 
Emotions detected:- "+str(compoundPositive))
data = [compoundPositive,compoundNegative,compoundNeutral]
#label = ["Postive: "+str(compoundPositive),"Negative-
"+str(compoundNegative),"Neutral- "+str(compoundNeutral)]
label = 
["Postive","Negative","Neutral"] 
colors = ("cyan","orange","grey")
wp = { 'linewidth' : 1, 'edgecolor' : 
"green" } explode=(0.0,0.2,0.4)
def func(pct, allvalues):
#print(pct)
absolute = math.ceil(pct / 
100*np.sum(allvalues)) return 
"{:.1f}%\n({:d})".format(pct, absolute)
fig = plt.figure(figsize =(10, 

7)) plt.pie(data, labels = 
label,
autopct = lambda pct: func(pct,data),

wedgeprops = 
wp, 
colors=colors, 
explode=explod
e, shadow=True,
textprops = dict(color = "magenta",size=11, 
weight="bold")) plt.title("Different Emotions Pie-Chart")
plt.show()
def scatterplot(x,y,z):
fig = plt.figure()
ax = fig.add_subplot(111, 
projection='3d') ax.scatter(x, y, z, c='r', 
marker='o') ax.set_xlabel('Negative 
score') ax.set_ylabel('Neutral score') 
ax.set_zlabel('Positive score')
plt.show()
if name == ' main ':
fp = open("processed.txt",'r',encoding='utf-8')
sentences,compoundScore,compoundScoreNeg, compoundScoreNeu, compoundScorePos 
=[],[],[],[],[] for line in fp:
line = " ".join(filter(lambda x:x, 
line.split('\n'))) sentences.append(line)
#print(sentence
s) fp.close()
fp = open("completeOriginal.txt",'w',encoding='utf-8')
fileData = open("data.txt",'w') # stored in order 
negative,neutral,positive analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
vs = analyzer.polarityScore(sentence) 
compoundScore.append(vs["compound"]) 
compoundScoreNeg.append(vs["neg"]) 
compoundScoreNeu.append(vs["neu"]) 
compoundScorePos.append(vs["pos"]) 
fileData.write(str(vs["neg"])+","+str(vs["neu"])+","+str(vs["pos"])+"\n") 
print("{:-<65} {}".format(sentence, str(vs))+"\n")
fp.write("{:-<65} {}".format(sentence, 
str(vs))+"\n") print(" ") 
fp.close()
fileData.close() 
#print(compoundScore) 
visualize(compoundScore)
scatterplot(compoundScoreNeg,compoundScoreNeu,compoundScore
Pos) print("\n\n Done!")

kMeans.py
import math 
import 
random
import numpy as 
np import pandas 
as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def kMeans(sentimentList):
# Parameters k = number of clusters, & max 
iterations k = 3
maxIter = 2500
# Randomly selecting three unique 
centers centroid = 
random.sample(sentimentList, k)
n = 
len(sentimentList) 
clusterLabels = [0] * 
n
# Creating a list length to store number of data belonging to each 
cluster length = []
while maxIter:
# Assigning label of nearest centroid to each 
point for i in range(n):
minDistance = math.dist(sentimentList[i], centroid[0]) # Euclidean 
distance clusterLabels[i] = 0
for j in range(1, k):
if math.dist(sentimentList[i], centroid[j]) < minDistance:# 
Euclidean minDistance = math.dist(sentimentList[i], 
centroid[j]) clusterLabels[i] = j
#Creating a list for new centroid 
points newCentroid = []
length= [0]*k 
for i in 
range(k):
newCentroid.append([0.0, 0.0,0.0])
# To find mean, first finding sum of data belonging to each 
cluster for i in range(n):
temp = clusterLabels[i] 
newCentroid[temp][0] += 
sentimentList[i][0] newCentroid[temp][1] 
+= sentimentList[i][1] 
newCentroid[temp][2] += 
sentimentList[i][2] length[temp] += 1
# Now dividing sum by length of respective 
cluster for i in range(k):
if length[i]==0:
continue
newCentroid[i][0] = newCentroid[i][0] / 
length[i] newCentroid[i][1] = 

newCentroid[i][1] / length[i] newCentroid[i][2] 
= newCentroid[i][2] / length[i]

# Assigning new centroids to original 
centroids centroid = newCentroid
maxIter -= 1
# Returning labels belonging to each entry 
return
clusterLabels,length[0],length[1],length[2]
def visual(len_c1,len_c2,len_c3):
print("Number of elements in Cluster-1 :-
"+str(len_c1)) print("Number of elements in 
Cluster-2 :- "+str(len_c2)) print("Number of 
elements in Cluster-3 :- "+str(len_c3))
#Pi - chart data
data = [len_c3,len_c1,len_c2]
#label = ["Cluster-3: "+str(len_c3),"Cluster-1: "+str(len_c1),"Cluster-2: 
"+str(len_c2)] label = ["Cluster-3","Cluster-1","Cluster-2"]
wp = { 'linewidth' : 0.1, 'edgecolor' : "green" } 
explode=(0.0,0.2,0.4)
colors = 
("cyan","orange","grey") def 
func(pct, allvalues):
#print(pct)
absolute = math.ceil(pct / 
100*np.sum(allvalues)) return 
"{:.1f}%\n({:d})".format(pct, absolute)
fig = plt.figure(figsize =(10, 
7)) plt.pie(data, labels = 
label,
autopct = lambda pct: 
func(pct,data), shadow = True,
wedgeprops = 
wp, explode = 
explode, colors =
colors,
textprops= dict(color ="magenta",size= 8, 
weight="bold")) plt.title("Different Emotions Pie-Chart")
plt.show()
def scatterplot(dataset):
colors= ['blue','green','red']
dataset['color']= 
dataset.Cluster.map({0:colors[0],1:colors[1],2:colors[2]}) fig = 
plt.figure(figsize=(26,7))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(dataset.Negative, dataset.Neutral, dataset.Positive, c=dataset.color)
ax.set_xlabel('Negative') 
ax.set_ylabel('Neutral') 
ax.set_zlabel('Positive') 
ax.set_title('ClusterGraph') plt.show()
plt.scatter(dataset.Negative, dataset.Positive, 
c=dataset.color) plt.xlabel('Negative')
plt.ylabel('Positive') 
plt.title('Cluster-plot Neg-Pos 
axis') plt.show()

plt.scatter( dataset.Neutral, dataset.Positive, 
c=dataset.color) plt.xlabel('Neutral')

plt.ylabel('Positive') 
plt.title('Cluster-plot Neu-Pos 
axis') plt.show()
plt.scatter(dataset.Negative, 
dataset.Neutral,c=dataset.color) plt.xlabel('Negative')
plt.ylabel('Neutral') 
plt.title('Cluster-plot Neg-Neu 
axis') plt.show()
if name == " main ":
fp = open("data.txt", 'r')
lines = fp.readlines() #in order of 
negative,neutral,positive fp.close()
sentimentList,neg,neu,pos = 
[],[],[],[] for line in lines:
temp = line.split(',')
temp[2] = 
temp[2].rstrip('\n') temp[0] 
= float(temp[0]) 
neg.append(float(temp[0])) 
temp[1] = float(temp[1]) 
neu.append(float(temp[1])) 
temp[2] = float(temp[2]) 
pos.append(float(temp[2])) 
sentimentList.append(tem
p)
result, len_c1, len_c2, len_c3 = 
kMeans(sentimentList) dataset = 
pd.DataFrame(list(zip(neg,neu,pos,result)),
columns= ['Negative','Neutral','Positive','Cluster']) 
visual(len_c1,len_c2,len_c3)
scatterplot(dataset)
