#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
TigerXMLCorpusReader.py
(c) 14 June, 2013  Will Roberts

Read TIGER corpus files in XML format.
'''

from nltk.corpus.reader.util    import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree                  import Tree, ParentedTree
from nltk.util                  import LazyConcatenation, LazyMap
from NegraCorpusReader          import Atom, NegraCorpusReader, _get_parsed_words_helper

class TigerXMLCorpusReader(XMLCorpusReader):
    '''
    Corpus reader for the TIGER XML corpus.
    '''

    def __init__(self, root, fileids):
        '''
        Creates a new TIGER XML corpus reader.

        Arguments:
        - `root`: the base directory for the TIGER corpus
        - `fileids`: the XML filename of the TIGER corpus
        '''
        super(TigerXMLCorpusReader, self).__init__(root, fileids)

    #==========================================================================
    # Data access methods
    #==========================================================================

    def words(self, fileids=None):
        '''
        Returns all of the words and punctuation symbols that were in
        text nodes.
        '''
        return LazyConcatenation(LazyMap(self._get_words,
                                         self._sentence_etrees(fileids)))

    def sents(self, fileids=None):
        '''
        Retrieves a list of unannotated sentences from the
        corpus.
        '''
        return LazyMap(self._get_words, self._sentence_etrees(fileids))

    def tagged_words(self, fileids=None):
        return LazyConcatenation(LazyMap(self._get_tagged_words,
                                         self._sentence_etrees(fileids)))

    def tagged_sents(self, fileids=None):
        return LazyMap(self._get_tagged_words, self._sentence_etrees(fileids))

    def lemmatised_words(self, fileids=None):
        '''
        Retrieve a list of lemmatised words. Words are encoded as
        tuples in C{(word, lemma)} form.

        @return: A list of words and their tuples.
        @rtype: C{list} of C{(word, lemma)}
        '''
        return LazyConcatenation(LazyMap(self._get_lemmatised_words,
                                         self._sentence_etrees(fileids)))

    def lemmatised_sents(self, fileids=None):
        '''
        Retrieve a list of sentences and the words' lemma. Words are
        encoded as tuples in C{(word, lemma)} form.

        @return: A list of sentences with words and their lemma.
        @rtype: C{list} of C{list} of C{(word, lemma)}
        '''
        return LazyMap(self._get_lemmatised_words,
                       self._sentence_etrees(fileids))

    def morphological_words(self, fileids=None):
        '''
        Retrieve a list of sentences with the words' morphological
        type.  Words are encoded as tuples in C{(word, morph)} form.

        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{(word, morph)}
        '''
        return LazyConcatenation(LazyMap(self._get_morphological_words,
                                         self._sentence_etrees(fileids)))

    def morphological_sents(self, fileids=None):
        '''
        Retrieve a list of sentences with the words' morphological
        type. Words are encoded as tuples in C{(word, morph)} form.

        @return: A list of sentences with words and their morphological type.
        @rtype: C{list} of C{list} of C{(word, morph)}
        '''
        return LazyMap(self._get_morphological_words,
                       self._sentence_etrees(fileids))

    def parsed_sents(self, fileids=None):
        '''
        Retrieve a list of parsed sents as L{Tree}. The tree
        leaves are bare strings containing the word, and are children
        to unary tree nodes containing the part of speech tag.

        @return: A list of sentence tree representations.
        @rtype: C{list} of L{Tree}
        '''
        return LazyMap(self._get_parsed_words,
                       self._sentence_etrees(fileids))

    def parsed_sents_morph(self, fileids = None, secedge_copy = True):
        '''
        Retrieve a list of parsed sents as L{ParentedTree} with
        morphological information stored in the tree leaves. The tree
        leaves are objects which act like bare strings containing the
        word, but with extra properties C{tag}, C{morph},
        C{grid_lineno} and C{parent}; the leaves are children to unary
        tree nodes containing the part of speech tag.

        @return: A list of sentence tree representations.
        @rtype: C{list} of L{ParentedTree}
        '''
        return LazyMap(lambda s: self._get_parsed_words_morph(s, secedge_copy),
                       self._sentence_etrees(fileids))

    #==========================================================================
    # Transforms
    #==========================================================================

    def _sentence_etrees(self, fileids=None):
        return concat([XMLCorpusView(fileid, '.*/s')
                       for fileid in self.abspaths(fileids)])

    def _get_lemmatised_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [(terminal.get('word'), terminal.get('lemma')) for
                terminal in graph.getiterator("t")]

    def _get_morphological_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [(terminal.get('word'), terminal.get('morph')) for
                terminal in graph.getiterator("t")]

    def _get_parsed_words(self, sentence_etree):
        '''
        Builds a parse tree of type C{Tree} from the grid. The tree
        leaves are bare strings containing the word, and are children
        to unary tree nodes containing the part of speech tag.

        @return: Return a tree representation of parsed words from the grid.
        @rtype: L{Tree}
        '''
        tokens = _sentence_etree_to_tokens(sentence_etree)
        return _get_parsed_words_helper(tokens,
                                        Tree,
                                        lambda l, t, n: t[NegraCorpusReader.WORDS],
                                        False)

    def _get_parsed_words_morph(self, sentence_etree, secedge_copy = True):
        '''
        Builds a parse tree of type C{ParentedTree} from the grid. The
        tree leaves are objects which act like bare strings containing
        the word, but with extra properties C{tag}, C{morph},
        C{grid_lineno} and C{parent}; the leaves are children to unary
        tree nodes containing the part of speech tag.

        @return: Return a tree representation of parsed words from the grid.
        @rtype: L{ParentedTree}
        '''
        tokens = _sentence_etree_to_tokens(sentence_etree)
        return _get_parsed_words_helper(
            tokens,
            ParentedTree,
            lambda l, t, n: Atom(word=t[NegraCorpusReader.WORDS],
                                 tag=t.get(NegraCorpusReader.POS, None),
                                 morph=t.get(NegraCorpusReader.MORPH, None),
                                 lemma=t.get(NegraCorpusReader.LEMMA, None),
                                 edge=t.get(NegraCorpusReader.EDGE, None),
                                 secedge=t.get(NegraCorpusReader.SECEDGE, None),
                                 comment=t.get(NegraCorpusReader.COMMENT, None),
                                 grid_lineno=l,
                                 parent=n),
            secedge_copy)

    def _get_tagged_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [(terminal.get('word'), terminal.get('pos')) for
                terminal in graph.getiterator("t")]

    def _get_words(self, sentence_etree):
        graph = sentence_etree.find('graph')
        return [terminal.get('word') for terminal in graph.getiterator("t")]

XML_ATTR_TO_NEGRA_COL_NAME_MAP = {
    'word':   NegraCorpusReader.WORDS,
    'lemma':  NegraCorpusReader.LEMMA,
    'pos':    NegraCorpusReader.POS,
    'morph':  NegraCorpusReader.MORPH,
    #'case':   NegraCorpusReader.CASE,
    #'number': NegraCorpusReader.NUMBER,
    #'gender': NegraCorpusReader.GENDER,
    #'person': NegraCorpusReader.PERSON,
    #'degree': NegraCorpusReader.DEGREE,
    #'tense':  NegraCorpusReader.TENSE,
    #'mood':   NegraCorpusReader.MOOD,
    }

def _rewrite_dict_keys(dict_items, key_name_map):
    '''
    Helper function to rename the keys of a dictionary.
    '''
    return ((key_name_map.get(k, k), v) for (k, v) in dict_items)

def _sentence_etree_to_tokens(sentence_etree):
    '''
    Helper function to transform an ElementTree element read from the
    TIGER XML corpus into the format used by the NegraCorpusReader.
    '''
    graph = sentence_etree.find('graph')
    id_to_tokens = {}
    tokens = []
    # insert terminals into tokens list in sentence order
    for idx, terminal in enumerate(graph.getiterator("t")):
        tokens.append(dict(_rewrite_dict_keys(terminal.items(),
                                              XML_ATTR_TO_NEGRA_COL_NAME_MAP)))
        tokens[-1].update(dict([
                    (NegraCorpusReader.EDGE,    None),
                    (NegraCorpusReader.PARENT,  None),
                    (NegraCorpusReader.SECEDGE, None),
                    (NegraCorpusReader.COMMENT, None)]))
        id_to_tokens[terminal.get('id')] = (idx, tokens[-1])
    # insert non-terminals into tokens list after the terminals
    num_terminals = len(tokens)
    non_terminal_words = []
    for idx, nonterminal in enumerate(graph.getiterator("nt")):
        idx += num_terminals
        tok_id = '#' + nonterminal.get('id').split('_')[1]
        non_terminal_words.append(tok_id[1:])
        tok = {}
        tok[NegraCorpusReader.WORDS]   = tok_id
        tok[NegraCorpusReader.LEMMA]   = '--'
        tok[NegraCorpusReader.POS]     = nonterminal.get('cat')
        tok[NegraCorpusReader.MORPH]   = '--'
        tok[NegraCorpusReader.EDGE]    = None
        tok[NegraCorpusReader.PARENT]  = 0
        tok[NegraCorpusReader.SECEDGE] = None
        tok[NegraCorpusReader.COMMENT] = None
        tokens.append(tok)
        id_to_tokens[nonterminal.get('id')] = (idx, tokens[-1])
    # renumber the non-terminals which have non-numeric names (e.g., VROOT)
    renumbered_non_terminal_value = [int(x) for x in non_terminal_words
                                        if x.isdigit()]
    if renumbered_non_terminal_value:
        renumbered_non_terminal_value = max(renumbered_non_terminal_value) + 1
    else:
        renumbered_non_terminal_value = 500
    for tok in tokens[num_terminals:]:
        if not tok[NegraCorpusReader.WORDS][1:].isdigit():
            new_name = '#{0}'.format(renumbered_non_terminal_value)
            tok[NegraCorpusReader.WORDS] = new_name
            renumbered_non_terminal_value += 1
    # connect terminals and non-terminals to their parents
    for nonterminal in graph.getiterator("nt"):
        parent_tok  = id_to_tokens[nonterminal.get('id')][1]
        parent_word = parent_tok[NegraCorpusReader.WORDS][1:]
        for edge in nonterminal.getiterator('edge'):
            tok = id_to_tokens[edge.get('idref')][1]
            tok[NegraCorpusReader.PARENT]  = parent_word
            tok[NegraCorpusReader.EDGE]    = edge.get('label')
        for secedge in nonterminal.getiterator('secedge'):
            tok = id_to_tokens[secedge.get('idref')][1]
            tok[NegraCorpusReader.COMMENT] = parent_word
            tok[NegraCorpusReader.SECEDGE] = secedge.get('label')
    # move the root nonterminal to the end of tokens
    last_idx    = len(tokens) - 1
    root_nt_idx = id_to_tokens[graph.get('root')][0]
    if last_idx != root_nt_idx:
        tokens[last_idx], tokens[root_nt_idx] = (tokens[root_nt_idx],
                                                 tokens[last_idx])
    return tokens
