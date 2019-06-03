
# example calls for running with different feature and models
#-----------------------------------------------------------
#
# make FEAT=ud2+ud3+ud4.trigram TRAIN=360 eval
#
# make UDTAG=3 FEAT=ud3.trigram TRAIN=100 eval
# make UDTAG=4 FEAT=ud4.bigram TRAIN=100 eval
# make UDTAG=4 FEAT=ud4.trigram TRAIN=100 eval
#
# make FEAT=unigram TRAIN=100 eval
# make FEAT=bigram TRAIN=100 eval
# make FEAT=trigram TRAIN=100 eval
#
# make FEAT=trigram MODEL=multiclass TRAIN=100 eval
#
# make FEAT=trigram TRAIN=100 LABEL=2 eval
# make FEAT=trigram TRAIN=100 LABEL=3 eval
# make FEAT=trigram TRAIN=100 LABEL=23 eval



## column with the label to be predicted
## - 2 = prosody
## - 3 = boundary ?
## - 23 = both combined into one label
LABEL = 2

## training data to be used (100 or 360)
TRAIN = 360

## training steps for estimating learning curves
STEPS = 0.01 0.05 0.1 0.5 1

## features to be used
FEAT = ud2+ud3+ud4.trigram

## UDTAG = field in UDpipe-annotated data to be used as feature
##  2 = surface form words
##  3 = lemma
##  4 = universal POS tag
##  5 = extended POS tag
##  6 = morphological features
UDTAG = 4

## model to be used and number of max iterations in training
# MODEL = multiclass
MODEL = multitron
MAXITER = 25


# UDPIPE = udpipe/src/udpipe
# MEGAM = megam

SYSTEM = ${shell uname -s}
BINDIR = bin/${SYSTEM}
MEGAM  = ${BINDIR}/megam
UDPIPE = ${BINDIR}/udpipe
UDPIPE_MODEL = english-ewt-ud-2.3-181115.udpipe


all: test.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval-no-NA


data: train_${TRAIN}.l${LABEL}.${FEAT} test.l${LABEL}.bigram
train: train_${TRAIN}.l${LABEL}.${FEAT}.${MODEL}
test: test.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict-no-NA
eval: test.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval-no-NA


## run with growing amounts of data
learning-curve:
	for s in ${STEPS}; do \
	  ${MAKE} TRAIN=360 FEAT=${FEAT}_$$s all; \
	done

run-learning-curves:
	make FEAT=trigram learning-curve
	make UDTAG=3 FEAT=ud3.trigram learning-curve
	make UDTAG=4 FEAT=ud4.trigram learning-curve
	make UDTAG=5 FEAT=ud5.trigram learning-curve
	make FEAT=ud2+ud3+ud4.trigram learning-curve
	make FEAT=ud2+l3.trigram TRAIN=360 learning-curve

## run all settings for predicting label 2
run:
	make FEAT=unigram LABEL=2 TRAIN=360 eval
	make FEAT=bigram LABEL=2 TRAIN=360 eval
	make FEAT=trigram LABEL=2 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.unigram LABEL=2 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.unigram LABEL=2 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.unigram LABEL=2 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.trigram LABEL=2 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.trigram LABEL=2 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.trigram LABEL=2 TRAIN=360 eval
	make FEAT=ud2+ud3+ud4.trigram TRAIN=360 LABEL=2 eval
	make FEAT=l3.unigram TRAIN=360 LABEL=2 eval
	make FEAT=l3.bigram TRAIN=360 LABEL=2 eval
	make FEAT=l3.trigram TRAIN=360 LABEL=2 eval
	make FEAT=ud2+l3.trigram TRAIN=360 LABEL=2 eval

## run all settings for predicting label 3
run3:
	make FEAT=unigram LABEL=3 TRAIN=360 eval
	make FEAT=bigram LABEL=3 TRAIN=360 eval
	make FEAT=trigram LABEL=3 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.trigram LABEL=3 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.trigram LABEL=3 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.trigram LABEL=3 TRAIN=360 eval
	make FEAT=ud2+ud3+ud4.trigram TRAIN=360 LABEL=3 eval

## run all settings for predicting label 2 and 3 together
run23:
	make FEAT=unigram LABEL=23 TRAIN=360 eval
	make FEAT=bigram LABEL=23 TRAIN=360 eval
	make FEAT=trigram LABEL=23 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.trigram LABEL=23 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.trigram LABEL=23 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.trigram LABEL=23 TRAIN=360 eval
	make FEAT=ud2+ud3+ud4.trigram TRAIN=360 LABEL=23 eval



%.l2: ../data/%.txt
	cut -f2 $<  > $@

%.l3: ../data/%.txt
	cut -f3 $<  > $@

## combine both labels
%.l23: ../data/%.txt
	cut -f2,3 $< | \
	awk '{ if ( $$1 == "NA"){print "NA"} else if ($$1 == "") {print ""} else {print $$1+3*$$2} }' > $@


## learning curves: growing amounts of data
## this only works for the 360 data (hard coded line counts here)
##
## line count of the train_360 data: 2494437

%_0.01: %
	head -24944 $< > $@

%_0.05: %
	head -124722 $< > $@

%_0.1: %
	head -249444 $< > $@

%_0.5: %
	head -1247219 $< > $@

%_1: %
	cp $< $@




## combined features

## word + lemma + UPOS trigrams
%.l${LABEL}.ud2+ud3+ud4.trigram:
	${MAKE} UDTAG=2 ${@:ud2+ud3+ud4.trigram=ud2.trigram}
	${MAKE} UDTAG=3 ${@:ud2+ud3+ud4.trigram=ud3.trigram}
	${MAKE} UDTAG=4 ${@:ud2+ud3+ud4.trigram=ud4.trigram}
	cut -f1 ${@:ud2+ud3+ud4.trigram=ud2.trigram} > $@.labels
	cut -f2 ${@:ud2+ud3+ud4.trigram=ud2.trigram} > $@.feat2
	cut -f2 ${@:ud2+ud3+ud4.trigram=ud3.trigram} > $@.feat3
	cut -f2 ${@:ud2+ud3+ud4.trigram=ud4.trigram} > $@.feat4
	paste -d ' ' $@.feat2 $@.feat3 $@.feat4 > $@.feat
	paste $@.labels $@.feat > $@
	rm -f $@.feat2 $@.feat3 $@.feat4 $@.feat $@.labels

## word + boundary label trigrams
%.l${LABEL}.ud2+l3.trigram:
	${MAKE} UDTAG=2 ${@:ud2+l3.trigram=ud2.trigram}
	${MAKE} ${@:ud2+l3.trigram=l3.trigram}
	cut -f1 ${@:ud2+l3.trigram=ud2.trigram} > $@.labels
	cut -f2 ${@:ud2+l3.trigram=ud2.trigram} > $@.feat2
	cut -f2 ${@:ud2+l3.trigram=l3.trigram} > $@.feat3
	paste -d ' ' $@.feat2 $@.feat3 > $@.feat
	paste $@.labels $@.feat > $@
	rm -f $@.feat2 $@.feat3 $@.feat $@.labels



# boundary labels as features

%.l2.l3.unigram: %.l2 %.l3
	paste $^ | sed 's/NA/0/' | egrep "^[0-9]\t" > $@

%.l2.l3.bigram: %.l3 %.l2 
	echo 'w0_START' > $@.tags1
	cat $< | sed 's/^ *$$/START/' | sed '$$d' | sed 's/^/w0_/' >> $@.tags1
	cat $< | sed 's/^/w1_/' > $@.tags2
	paste -d ' ' $@.tags1 $@.tags2  > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.tags1 $@.tags2 $@.feats

%.l2.l3.trigram: %.l3 %.l2
	echo 'w0_START' > $@.tags1
	grep -v '^# ' $< | sed 's/^ *$$/START/' | sed '$$d' | sed 's/^/w0_/' >> $@.tags1
	grep -v '^# ' $< | sed 's/^/w1_/' > $@.tags2
	grep -v '^# ' $< | sed 's/^ *$$/END/' | tail -n +2 | sed 's/^/w2_/' > $@.tags3
	echo 'w2_END' >> $@.tags3
	paste -d ' ' $@.tags1 $@.tags2 $@.tags3 > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.tags1 $@.tags2 $@.tags3 $@.feats



## UD tag n-grams

.PRECIOUS: %.tagged
%.tagged: ../data/%.txt
	cut -f1 $< | ${UDPIPE} --tag english-ewt-ud-2.3-181115.udpipe --input vertical > $@

%.l${LABEL}.ud${UDTAG}.unigram: %.tagged %.l${LABEL}
	grep -v '^# ' $< | cut -f${UDTAG} > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.feats

%.l${LABEL}.ud${UDTAG}.bigram: %.tagged %.l${LABEL}
	echo 'w0_START' > $@.tags1
	grep -v '^# ' $< | cut -f${UDTAG} | sed 's/^ *$$/START/' | sed '$$d' | sed 's/^/w0_/' >> $@.tags1
	grep -v '^# ' $< | cut -f${UDTAG} | sed 's/^/w1_/' > $@.tags2
	paste -d ' ' $@.tags1 $@.tags2  > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.tags1 $@.tags2 $@.feats

%.l${LABEL}.ud${UDTAG}.trigram: %.tagged %.l${LABEL}
	echo 'w0_START' > $@.tags1
	grep -v '^# ' $< | cut -f${UDTAG} | sed 's/^ *$$/START/' | sed '$$d' | sed 's/^/w0_/' >> $@.tags1
	grep -v '^# ' $< | cut -f${UDTAG} | sed 's/^/w1_/' > $@.tags2
	grep -v '^# ' $< | cut -f${UDTAG} | sed 's/^ *$$/END/' | tail -n +2 | sed 's/^/w2_/' > $@.tags3
	echo 'w2_END' >> $@.tags3
	paste -d ' ' $@.tags1 $@.tags2 $@.tags3 > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.tags1 $@.tags2 $@.tags3 $@.feats




## word n-grams

%.l${LABEL}.unigram: ../data/%.txt %.l${LABEL}
	cut -f1 $< > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.feats

%.l${LABEL}.bigram: ../data/%.txt %.l${LABEL}
	echo 'w0_START' > $@.words1
	cut -f1 $< | sed 's/^ *$$/START/' | sed '$$d' | sed 's/^/w0_/' >> $@.words1
	cut -f1 $< | sed 's/^/w1_/' > $@.words2
	paste -d ' ' $@.words1 $@.words2  > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.words1 $@.words2 $@.feats

%.l${LABEL}.trigram: ../data/%.txt %.l${LABEL}
	echo 'w0_START' > $@.words1
	cut -f1 $< | sed 's/^ *$$/START/' | sed '$$d' | sed 's/^/w0_/' >> $@.words1
	cut -f1 $< | sed 's/^/w1_/' > $@.words2
	cut -f1 $< | sed 's/^ *$$/END/' | tail -n +2 | sed 's/^/w2_/' > $@.words3
	echo 'w2_END' >> $@.words3
	paste -d ' ' $@.words1 $@.words2 $@.words3 > $@.feats
	paste ${word 2,$^} $@.feats | sed 's/NA/0/' | egrep "^[0-9]\t" > $@
	rm -f $@.words1 $@.words2 $@.words3 $@.feats


%.multitron: %
	${MEGAM} -maxi ${MAXITER} multitron $< > $@

%.multiclass: %
	${MEGAM} -maxi ${MAXITER} multiclass $< > $@

%.binary: %
	sed 's/^[2-9]/1/' < $< > $@.train
	${MEGAM} -maxi ${MAXITER} binary $@.train > $@
	rm -f $@.train


%.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict: %.l${LABEL}.${FEAT} train_${TRAIN}.l${LABEL}.${FEAT}.${MODEL} ../data/%.txt
	${MEGAM} -predict ${word 2,$^} ${MODEL} $< > $@.raw
	cut -f1 ${word 3,$^} | grep . > $@.words
	paste -d ' ' $@.words $@.raw | sed 's/^[[:punct:]].*$$/. NA/' | cut -f2 -d' ' > $@
	rm -f $@.raw $@.words

%.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval: %.l${LABEL} %.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict
ifeq (${MODEL},binary)
	perl eval.pl -b $< < ${word 2,$^} > $@
else
	perl eval.pl $< < ${word 2,$^} > $@
endif

## don't post-process to get NA labels
%.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict-no-NA: %.l${LABEL}.${FEAT} train_${TRAIN}.l${LABEL}.${FEAT}.${MODEL} ../data/%.txt
	${MEGAM} -predict ${word 2,$^} ${MODEL} $< > $@.raw
	cut -f1 ${word 3,$^} | grep . > $@.words
	paste -d ' ' $@.words $@.raw | cut -f2 -d' ' > $@
	rm -f $@.raw $@.words

%.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval-no-NA: %.l${LABEL} %.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict-no-NA
ifeq (${MODEL},binary)
	perl eval.pl -b -n $< < ${word 2,$^} > $@
else
	perl eval.pl -n $< < ${word 2,$^} > $@
endif


GOLDPREDICT = ${patsubst %.predict,%.gold+predict,${wildcard *.predict}}
goldpredict: ${GOLDPREDICT}


%.gold+predict: %.predict ../data/test.txt
	grep . ${word 2,$^} | cut -f1,${LABEL} > $@.gold
	cut -f1 $< > $@.sys
	paste -d ' ' $@.gold $@.sys > $@
	rm -f $@.gold $@.sys


EVAL2 = ${patsubst %.predict,%.eval2,${wildcard *.predict}}
eval2: ${EVAL2}


%.eval2: test.l${LABEL} %.predict
	perl eval2.pl $< < ${word 2,$^} > $@
