
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


all: test.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval


data: train_${TRAIN}.l${LABEL}.${FEAT} test.l${LABEL}.bigram
train: train_${TRAIN}.l${LABEL}.${FEAT}.${MODEL}
test: test.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict
eval: test.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval


run:
	make FEAT=unigram LABEL=2 TRAIN=360 eval
	make FEAT=bigram LABEL=2 TRAIN=360 eval
	make FEAT=trigram LABEL=2 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.trigram LABEL=2 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.trigram LABEL=2 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.trigram LABEL=2 TRAIN=360 eval
	make FEAT=unigram LABEL=3 TRAIN=360 eval
	make FEAT=bigram LABEL=3 TRAIN=360 eval
	make FEAT=trigram LABEL=3 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.trigram LABEL=3 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.trigram LABEL=3 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.trigram LABEL=3 TRAIN=360 eval
	make FEAT=unigram LABEL=23 TRAIN=360 eval
	make FEAT=bigram LABEL=23 TRAIN=360 eval
	make FEAT=trigram LABEL=23 TRAIN=360 eval
	make UDTAG=3 FEAT=ud3.trigram LABEL=23 TRAIN=360 eval
	make UDTAG=4 FEAT=ud4.trigram LABEL=23 TRAIN=360 eval
	make UDTAG=5 FEAT=ud5.trigram LABEL=23 TRAIN=360 eval
	make FEAT=ud2+ud3+ud4.trigram TRAIN=360 LABEL=2 eval
	make FEAT=ud2+ud3+ud4.trigram TRAIN=360 LABEL=3 eval
	make FEAT=ud2+ud3+ud4.trigram TRAIN=360 LABEL=23 eval




%.l2: ../data/%.txt
	cut -f2 $<  > $@

%.l3: ../data/%.txt
	cut -f3 $<  > $@

## combine both labels
%.l23: ../data/%.txt
	cut -f2,3 $< | \
	awk '{ if ( $$1 == "NA"){print "NA"} else if ($$1 == "") {print ""} else {print $$1+3*$$2} }' > $@

## combined features

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


%.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict: %.l${LABEL}.${FEAT} train_${TRAIN}.l${LABEL}.${FEAT}.${MODEL} ../data/%.txt
	${MEGAM} -predict ${word 2,$^} ${MODEL} $< > $@.raw
	cut -f1 ${word 3,$^} | grep . > $@.words
	paste -d ' ' $@.words $@.raw | sed 's/^[[:punct:]].*$$/. NA/' | cut -f2 -d' ' > $@
	rm -f $@.raw $@.words

%.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.eval: %.l${LABEL} %.${TRAIN}.l${LABEL}.${FEAT}.${MODEL}.predict
	perl eval.pl $< < ${word 2,$^} > $@
