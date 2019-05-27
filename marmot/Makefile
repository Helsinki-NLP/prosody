
# marmot: http://cistern.cis.lmu.de/marmot/


## column with the label to be predicted
## - 1 = prosody
## - 2 = boundary ?
LABEL = 1

## training data to be used (100 or 360)
TRAIN = 360
# TRAIN = 100


MARMOT = marmot-2015-10-22.jar

all: test.${TRAIN}.l${LABEL}.eval

train: train_${TRAIN}.l${LABEL}.marmot
test: test.${TRAIN}.l${LABEL}.predict
eval: test.${TRAIN}.l${LABEL}.eval


%.l1: ../data/%.txt
	cut -f2 $<  > $@

%.l2: ../data/%.txt
	cut -f3 $<  > $@


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
