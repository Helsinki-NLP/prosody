
# marmot: http://cistern.cis.lmu.de/marmot/


## column with the label to be predicted
## - 1 = prosody
## - 2 = boundary ?
LABEL = 1

## training data to be used (100 or 360)
TRAIN = 360
# TRAIN = 100

## training steps for estimating learning curves
STEPS = 0.01 0.05 0.1 0.5 1


MARMOT = marmot-2015-10-22.jar

all: test.${TRAIN}.l${LABEL}.eval

train: train_${TRAIN}.l${LABEL}.marmot
test: test.${TRAIN}.l${LABEL}.predict
eval: test.${TRAIN}.l${LABEL}.eval

learning-curve:
	for s in ${STEPS}; do \
	  ${MAKE} TRAIN=360_$$s all; \
	done

%.l1: ../data/%.txt
	cut -f2 $<  > $@

%.l2: ../data/%.txt
	cut -f3 $<  > $@


## learning curves: growing amounts of data
## line count of the train_360 data: 2494437

../data/train_360_0.01.txt: ../data/train_360.txt
	head -24944 $< > $@

../data/train_360_0.05.txt: ../data/train_360.txt
	head -124722 $< > $@

../data/train_360_0.1.txt: ../data/train_360.txt
	head -249444 $< > $@

../data/train_360_0.5.txt: ../data/train_360.txt
	head -1247219 $< > $@

../data/train_360_1.txt: ../data/train_360.txt
	cp $< $@


# /System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/java

%.l${LABEL}.marmot: ../data/%.txt
	java -Xmx5G -cp ${MARMOT} marmot.morph.cmd.Trainer \
		-train-file form-index=0,tag-index=${LABEL},$< \
		-tag-morph false \
		-model-file $@

%.${TRAIN}.l${LABEL}.predict: ../data/%.txt train_${TRAIN}.l${LABEL}.marmot
	java -cp ${MARMOT} marmot.morph.cmd.Annotator \
		--model-file ${word 2,$^} \
		--test-file form-index=0,$< \
		--pred-file $@.tmp
	cut -f6 $@.tmp > $@
	rm -f $@.tmp

%.${TRAIN}.l${LABEL}.eval: %.l${LABEL} %.${TRAIN}.l${LABEL}.predict
	perl ../megam/eval.pl -n $< < ${word 2,$^} > $@
