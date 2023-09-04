# eBay 2023 ML Competition

# The Problem

The problem we invite you to consider for this year is to build a model that can accurately extract and label the named entities in the dataset of item titles on eBay. Named Entities are the semantic strings/words/phrases that refer to people, brands, organizations, locations, styles, materials, patterns, product names, units of measure, clothing sizes, etc.

Named Entity Recognition (NER) is the machine learning process of automatic labeling and extracting important named entities in a text that carry a particular meaning. In e-commerce, NER is used to process listing or product titles and descriptions, queries, and reviews, or wherever extraction of important data from raw text is desired.

At eBay, we apply NER in a variety of applications, in particular for extracting aspects from listings (seller-facing context), and from search queries (buyer-facing context). In both of these contexts NER plays a crucial role to bridge unstructured text data to structured data. This challenge focuses on extraction from listings.

# The Challenge

Named entity recognition (NER) is a fundamental task in Natural Language Processing (NLP) and one of the first stages in many language understanding tasks. It has drawn research attention for a few decades, and its importance has been well recognized in both academia and industry. **While NER is applied in many different settings, for this challenge, we will only be using eBay listing titles for NER**. A few examples of NER labeling of listing titles are shown below (these examples are in English to illustrate the concept, the challenge data will have German language listing titles).

![Screen Shot 2023-08-30 at 8 34 34 PM](https://github.com/xavajk/ner/assets/95323308/1adfc637-13e5-4410-b2b5-39ae03576e06)

The extracted entities are also called aspects, and an aspect consists of the aspect name (“Brand name” for the first aspect in the last example above) and the aspect value (“NYX” for the same aspect in the same example above). The objective of this challenge then is to extract and label the aspects in the dataset of item titles listed on eBay. Not all titles have all aspects, and figuring out which aspect is present for a given title is part of the challenge.

# The Data

The data set consists of 10 million randomly selected unlabeled item titles from eBay Germany, all of which are from “Athletic Shoes” categories. Among these item titles there will be 10,000 labeled item titles (“labeled” means the aspects have been extracted). There will also be an annexure document provided that describes the dataset. Finally, we will provide the set of aspect names that should be extracted from each item title (as stated before, not all titles have all aspects). Each item title will have a unique identifier (a record number).

The 10,000 labeled item titles will be split into three groups:

1. Training set (5,000 records)
2. Quiz set (2,500 records)
3. Test set (2,500 records)

The 10 million unlabeled title set and the training set is intended for participants to build their models/prediction system. The actual aspects will be provided for each item title in the training set, along with the item title record number to link the aspects to the title.

The quiz data is used for leaderboard scoring. The aspects will not be provided for the quiz set. The precise set of records belonging to the quiz set will not be provided, instead a superset of 25,000 record identifiers will be provided (meaning the 2,500 quiz titles are a subset of this superset).

The full 10 million title dataset, along with the training and the quiz sets are expected to be released when the competition opens on **May 1st, 2023**.

The test set is used as one of the main factors to determine the winner, and it will only be distributed to the top-scoring teams after the competition closes on EvalAI on **November 28th, 2023**. Participants are expected to have their models/prediction system ready to score and submit their predictions within a few days of being contacted by eBay after the competition closes on EvalAI. Similar to the quiz set, the values for the aspect column will **not** be provided for the test set, and only a superset of 25,000 record numbers will be provided.

# NER with Spacy

# NER with NLTK

# Custom NER Approaches

## What is NER?

NER is a sequence-tagging task where the contextual meaning of words is fetched by using word embeddings. The NER model is used for information extraction and classifying named entities from unstructured text into pre-defined categories. Named entities are real-world objects such as a person's name, location, landmark, etc. It plays a key role in information extraction from documents and conversational data. In conversational agents, NER is used for entity recognition and information extraction to retrieve pieces of information like date, location, email, phone number, etc. The context plays a crucial role in NLP tasks, including NER. Traditional word embeddings like GLOVE, fasttext, and Word2Vec assign only one representation per word, while in reality, different words have different meanings depending on the context. 

## [CNN + BLSTM + CRF](https://arxiv.org/pdf/1603.01354.pdf) with [GloVe German Wikipedia Word Embeddings](https://www.deepset.ai/german-word-embeddings)

Ma and Hovy present a sequence labeling system that incorporates word context into tagging via the use of word embeddings ******and****** a Bi-directional LSTM (BLSTM). Intially, character-level represenations of the sequence words are generated using a CNN, the output of the CNN is then concatenated with the original word embeddings. For the eBay challengs, since the item titles are in german, the German Wikipedia corpus and embeddings will be used. The concatenated word and character representations are fed into the BLSTM to model context information of each word, including them into a CRF (Conditional Random Field) to jointly decode labels for the whole title.

The architecture presented in the paper implies the use of [BIO2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) annotations for sequence tokens, however the eBay challengs gives its own set of tags that should be identified. This is another difference that must be addressed.

- tiktoken tokenizer opposed to character level?

## [ELMo Word Embeddings](https://medium.com/saarthi-ai/how-to-make-your-own-ner-model-with-contexual-word-embeddings-5086276e04a0)

**[ELMO](https://allennlp.org/elmo)** understands both the meaning of the words and the context in which they are found, as opposed to GLOVE embeddings, which only capture the meaning of the words, and are unaware of the context.

![Screen Shot 2023-08-31 at 9 46 33 PM](https://github.com/xavajk/ner/assets/95323308/bcf06b02-a801-4856-8ca3-9f358fd85263)

ELMO assigns embeddings to words based on the contexts in which they are used — to both capture the word meaning in that context as well as to fetch other contextual information.

Instead of using a fixed embedding for each word like in GLOVE, ELMo looks at the entire sentence before assigning each word an embedding. It uses a [bi-directional LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) trained on specific tasks to create those word embeddings.

ELMO provided a significant step towards pre-training in the context of Natural Language Processing (NLP). The Elmo LSTM can be trained on a massive dataset in any language to make custom language models , and then be re-used as a component in other models that are tasked with NLU (Natural Language Understanding).
