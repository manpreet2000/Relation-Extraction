# Relation-Extraction [![Code of Conduct](https://img.shields.io/badge/code%20of-conduct-ff69b4.svg?style=flat)](https://github.com/tterb/hyde/blob/master/docs/CODE_OF_CONDUCT.md) 
All type of Relation Extraction Data has been added to stack, peek it out.
<br>
![Gify](https://media.giphy.com/media/jPBh54kLtXK2VAFq8L/giphy.gif)
<br>
List of Relation Extraction (Named Entity, CNN, DRNN, Distinct Supervision, etc) work located here 🤔 and totally motivated by [This guy](https://github.com/roomylee/awesome-relation-extraction).

## Papers

* Matching the Blanks: Distributional Similarity for Relation Learning [[paper]](https://paperswithcode.com/paper/matching-the-blanks-distributional-similarity)  [[code]](https://github.com/plkmo/BERT-Relation-Extraction)
     * Method : BERTEM+MTB  
* Coreferential Reasoning Learning for Language Representation [[paper]](https://paperswithcode.com/paper/coreferential-reasoning-learning-for-language)  [[code]](https://github.com/thunlp/KernelGAT)
     * Method : CorefRoBERTaLarge 
* Downstream Model Design of Pre-trained Language Model for Relation Extraction Task [[paper]](https://paperswithcode.com/paper/downstream-model-design-of-pre-trained)  [[code]](https://github.com/slczgwh/REDN)
     * Method : REDN 
* RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information [[paper]](https://paperswithcode.com/paper/reside-improving-distantly-supervised-neural)  [[code]](https://github.com/malllabiisc/RESIDE)
     * Method : RESIDE 
* Classifying Relations by Ranking with Convolutional Neural Networks [[paper]](https://arxiv.org/abs/1504.06580)  [[code]](https://github.com/pratapbhanu/CRCNN)
     * Method : CRNN
* MIT at SemEval-2017 Task 10: Relation Extraction with Convolutional Neural Networks [[paper]](https://aclanthology.info/pdf/S/S17/S17-2171.pdf)
	   * Method : CNN
* End-to-end Named Entity Recognition and Relation Extraction using Pre-trained Language Models [[paper]](https://paperswithcode.com/paper/end-to-end-named-entity-recognition-and-1)  [[code]](https://github.com/bowang-lab/joint-ner-and-re)
     * Method : NER
* Entity, Relation, and Event Extraction with Contextualized Span Representations [[paper]](https://paperswithcode.com/paper/entity-relation-and-event-extraction-with)  [[code]](https://github.com/diffbot/knowledge-net)
* Relation Extraction Using Distant Supervision: a Survey [[paper]](https://exascale.info/assets/pdf/smirnova2019acmcsur.pdf)  
* Global Relation Embedding for Relation Extraction [[paper]](https://www.aclweb.org/anthology/N18-1075.pdf)  [[code]](https://github.com/bowang-lab/joint-ner-and-re)     
* GREG: A Global Level Relation Extraction with Knowledge Graph Embedding [[paper]](https://www.researchgate.net/publication/339188625_GREG_A_Global_Level_Relation_Extraction_with_Knowledge_Graph_Embedding)  
     * Method : CNN
* Relation Extraction with Explanation   
     * DOI :  10.18653/v1/2020.acl-main.579
* End-to-End Relation Extraction using LSTMs on Sequences and Tree Structure [[paper]](https://arxiv.org/abs/1601.00770)
     * Method : LSTM/RNN
* Semantic Relation Classification via Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity Typing [[paper]](https://arxiv.org/abs/1901.08163) [[code]](https://github.com/roomylee/entity-aware-relation-classification)
* Classifying Relations via Long Short Term Memory Networks along Shortest Dependency Path [[paper]](https://arxiv.org/abs/1508.03720) [[code]](https://github.com/Sshanu/Relation-Classification)
* Semantic Compositionality through Recursive Matrix-Vector Spaces [[paper]](http://aclweb.org/anthology/D12-1110) [[code]](https://github.com/pratapbhanu/MVRNN)
     * Method : RNN
* Distant supervision for relation extraction without labeled data [[paper]](https://web.stanford.edu/~jurafsky/mintz.pdf) [[review]](https://github.com/roomylee/paper-review/blob/master/relation_extraction/Distant_supervision_for_relation_extraction_without_labeled_data/review.md)
* Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations [[paper]](http://www.aclweb.org/anthology/P11-1055) [[code]](http://aiweb.cs.washington.edu/ai/raphaelh/mr/)
* Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks [[paper]](https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf) [[code]](https://github.com/may-/cnn-re-tf)
* Hierarchical Relation Extraction with Coarse-to-Fine Grained Attention[[paper]](https://aclweb.org/anthology/D18-1247)[[code]](https://github.com/thunlp/HNRE)
* SpanBERT: Improving pre-training by representing and predicting spans [[paper]](https://arxiv.org/pdf/1907.10529.pdf) [[code]](https://github.com/facebookresearch/SpanBERT)

## Datasets
### Kowledge Graphs
* Data Dumps [[download]](https://developers.google.com/freebase)
* Wiki Links Data [[download]](http://www.iesl.cs.umass.edu/data/data-wiki-links)
### Other Dataset
* TACRED: The TAC Relation Extraction Dataset 
    [[paper]](https://www.aclweb.org/anthology/D17-1004.pdf) 
    [[Website]](https://nlp.stanford.edu/projects/tacred/) 
    [[download]](https://catalog.ldc.upenn.edu/LDC2018T24)
* FewRel: Few-Shot Relation Classification Dataset [[paper]](https://arxiv.org/abs/1810.10147) [[Website]](http://zhuhao.me/fewrel)
	* This dataset is a supervised few-shot relation classification dataset. The corpus is Wikipedia and the knowledge base used to annotate the corpus is Wikidata.
### Technologies
* grakn ai [[Website]](https://grakn.ai/)
* Amazon Neptune [[Website]](https://aws.amazon.com/neptune/)
* Neo4j [[Website]](https://neo4j.com/)
* Blazegraph [[Website]](https://blazegraph.com/)

## Videos and Lectures
* [Stanford University: CS124](https://web.stanford.edu/class/cs124/), Dan Jurafsky
	* (Video) [Week 5: Relation Extraction and Question](https://www.youtube.com/watch?v=5SUzf6252_0&list=PLaZQkZp6WhWyszpcteV4LFgJ8lQJ5WIxK&ab_channel=FromLanguagestoInformation)
* [Washington University: CSE517](https://courses.cs.washington.edu/courses/cse517/), Luke Zettlemoyer
	* (Slide) [Relation Extraction 1](https://courses.cs.washington.edu/courses/cse517/13wi/slides/cse517wi13-RelationExtraction.pdf)
	* (Slide) [Relation Extraction 2](https://courses.cs.washington.edu/courses/cse517/13wi/slides/cse517wi13-RelationExtractionII.pdf)
* [New York University: CSCI-GA.2590](https://cs.nyu.edu/courses/spring17/CSCI-GA.2590-001/), Ralph Grishman
	* (Slide) [Relation Extraction: Rule-based Approaches](https://cs.nyu.edu/courses/spring17/CSCI-GA.2590-001/DependencyPaths.pdf)
* [Michigan University: Coursera](https://ai.umich.edu/portfolio/natural-language-processing/), Dragomir R. Radev
	* (Video) [Lecture 48: Relation Extraction](https://www.youtube.com/watch?v=TbrlRei_0h8&ab_channel=ArtificialIntelligence-AllinOne)
* [Virginia University: CS6501-NLP](http://web.cs.ucla.edu/~kwchang/teaching/NLP16/), Kai-Wei Chang
	* (Slide) [Lecture 24: Relation Extraction](http://web.cs.ucla.edu/~kwchang/teaching/NLP16/slides/24-relation.pdf)
This section has been copied from [This super repo]((https://github.com/roomylee/awesome-relation-extraction))


**Contributions**: Any type of contributions are accepted 
