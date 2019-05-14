#!/bin/bash

# Create folder structure
echo -e "\nCreating the folder structure...\n"

mkdir embeddings
echo -e "Done!"

# Download and unzip GloVe word embeddings
echo -e "\nDownloading and unzipping Glove 840B 300D to .vector_cache\n"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -a glove.840B.300d.zip -d .vector_cache/
rm -f glove.840B.300d.zip
echo -e "\nDone!"

